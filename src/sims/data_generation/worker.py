import errno
import glob
import json
import multiprocessing as mp
import os
import queue
import random
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Union, Dict, Any, Type, Optional

import numpy as np
import torch
from allenact.base_abstractions.sensor import SensorSuite
from allenact.utils.misc_utils import md5_hash_str_as_int
from allenact.utils.tensor_utils import batch_observations, to_device_recursively
from filelock import FileLock
from omegaconf import OmegaConf
from setproctitle import setproctitle as ptitle
from tqdm import tqdm

from sims.data_generation.datagen_utils import (
    write_config_file,
    parse_queue_message,
    save_trajectories,
    format_exception,
    print_error,
)
from sims.data_generation.path_planners import PathPlanner
from sims.data_generation.queues import FromToQueue, QueueMessage
from sims.tasks import BaseTaskSampler
from sims.utils.data_generation_utils.exception_utils import (
    PlannerFailedException,
    HouseInvalidForTaskException,
    TaskSamplerInInvalidStateError,
    IrrecoverablePlannerFailureDueToHouseException,
    TaskDifficultyIncorrectException,
)


def increment_metrics_json(
    metrics_json_dir: Optional[str],
    success: int = 0,
    failure: int = 0,
    crashed: int = 0,
    already_processed: int = 0,
    worker_ind: int = 0,
    planner_failures: int = 0,
    planner_successes: int = 0,
):
    if metrics_json_dir is None:
        return

    os.makedirs(metrics_json_dir, exist_ok=True)
    path = os.path.join(metrics_json_dir, "metrics.json")
    with FileLock(path + ".lock"):
        if os.path.exists(path):
            with open(path, "r") as f:
                metrics = json.load(f)
                if "already_processed" not in metrics:
                    metrics["already_processed"] = 0
                if "planner_failures" not in metrics:
                    metrics["planner_failures"] = 0
                if "planner_successes" not in metrics:
                    metrics["planner_successes"] = 0
        else:
            metrics = dict(
                success=0,
                failure=0,
                crashed=0,
                already_processed=0,
                planner_failures=0,
                planner_successes=0,
            )

        metrics["success"] += success
        metrics["failure"] += failure
        metrics["crashed"] += crashed
        metrics["already_processed"] += already_processed
        metrics["planner_successes"] += planner_successes
        metrics["planner_failures"] += planner_failures

        with open(path, "w") as f:
            json.dump(metrics, f)


def save_logging_info(
    task_type_str: str,
    logging_save_dir: Optional[str],
    time_taken: float,
    split: str,
    house_index: int,
    repeats: int,
    num_saved_trajectories: int,
    retries: int,
):
    if logging_save_dir is None:
        return

    keys = [
        "task_type_str",
        "time_taken",
        "split",
        "house_index",
        "repeats",
        "num_saved_trajectories",
        "retries",
    ]

    os.makedirs(logging_save_dir, exist_ok=True)
    path = os.path.join(logging_save_dir, "logs.tsv")
    with FileLock(path + ".lock"):
        if not os.path.exists(path):
            with open(path, "w") as f:
                f.write("\t".join(keys) + "\n")

        def to_str(inp):
            if isinstance(inp, float):
                return f"{inp:.1f}"
            else:
                return str(inp)

        with open(path, "a") as f:
            kwargs = locals()
            f.write("\t".join(to_str(kwargs[k]) for k in keys) + "\n")


def _remove_incomplete_save_dir(save_dir: str) -> None:
    """Remove an incomplete house directory, including an ENOTEMPTY fallback."""
    path = Path(save_dir)
    if not path.exists() and not path.is_symlink():
        return
    if path.is_symlink() or path.is_file():
        path.unlink()
        return

    try:
        shutil.rmtree(path)
    except OSError as error:
        if error.errno != errno.ENOTEMPTY:
            raise

        # Some network filesystems have transiently returned ENOTEMPTY from
        # rmtree. Delete the actual children of ``save_dir`` and retry the root.
        for child in path.iterdir():
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child)
            else:
                child.unlink()
        path.rmdir()


def offline_dataset_worker(
    constants: OmegaConf,
    task_sampler_type: Type[BaseTaskSampler],
    task_sampler_kwargs: Dict[str, Any],
    path_planner: PathPlanner,
    worker_ind: int,
    lor_queue: FromToQueue,
    device: Optional[Union[torch.device, int]],
    expected_split: str,
    top_level_save_dir: str,
    metrics_json_dir: Optional[str],
    is_alive_queue: Optional[mp.Queue] = None,
    pbar: Optional[tqdm] = None,
):
    ptitle(f"Offline Dataset Worker {worker_ind}")
    print(f"Starting worker {worker_ind}", flush=True)

    if device is None:
        device = -1

    write_config_file(cfg=constants, top_level_save_dir=top_level_save_dir)
    task_sampler: Optional[BaseTaskSampler] = (
        None  # Initialized only when needed below, faster when running scripts locally
    )
    message: Optional[QueueMessage] = None

    # An irrecoverable failure means generation must stop for the current house
    # because tasks cannot be sampled or the planner cannot produce a successful
    # plan. Repeated failures usually indicate that AI2-THOR or the task sampler
    # has entered a bad state, so the worker exits.
    # TODO: Restart the worker instead.
    num_sequential_irrecoverable_failures = 0
    max_allowed_sequential_irrecoverable_failures = 5

    # Consecutive sampler or planner failures allowed before the house is
    # considered irrecoverable.
    max_allowed_sequential_task_sampler_failures = 10
    max_allowed_sequential_planner_failures = 10
    try:
        while True:
            message = lor_queue.get(timeout=2)
            house_processing_start_time = time.time()
            trajectories_to_generate_info = parse_queue_message(
                message, expected_split=expected_split
            )

            split = trajectories_to_generate_info["split"]
            house_index = trajectories_to_generate_info["house_index"]
            repeats = trajectories_to_generate_info["repeats"]

            def save_logging_info_for_house(num_saved_trajectories: int, retries: int):
                save_logging_info(
                    task_type_str=task_sampler_type.task_type_str,
                    logging_save_dir=metrics_json_dir,
                    time_taken=time.time() - house_processing_start_time,
                    split=split,
                    house_index=house_index,
                    repeats=repeats,
                    num_saved_trajectories=num_saved_trajectories,
                    retries=retries,
                )

            def update_pbar(n=1):
                if pbar is not None:
                    pbar.update(n=n)

            split_save_dir = os.path.join(top_level_save_dir, split)
            os.makedirs(split_save_dir, exist_ok=True)

            traj_id_without_repeat = f"split_{split}__house_{house_index:06d}"

            cur_save_dir = os.path.join(split_save_dir, f"{house_index:06d}")

            success_save_path = os.path.join(cur_save_dir, "success.txt")

            if os.path.exists(success_save_path):
                print(
                    f"[Worker {worker_ind}] skipping {traj_id_without_repeat} as it was already processed. Path: {success_save_path}",
                    flush=True,
                )
                increment_metrics_json(
                    metrics_json_dir=metrics_json_dir,
                    already_processed=1,
                    worker_ind=worker_ind,
                )
                lor_queue.mark_complete(message)
                update_pbar()
                continue

            if os.path.exists(cur_save_dir):
                # If the data wasn't saved successfully previously, delete what we have so far
                print(
                    f"[Worker {worker_ind}] {cur_save_dir} ALREADY EXISTS. Attempting to delete.",
                    flush=True,
                )
                try:
                    _remove_incomplete_save_dir(cur_save_dir)
                except OSError:
                    print_error(
                        f"[Worker {worker_ind}] Unable to delete an incomplete "
                        f"directory at {cur_save_dir}. Skipping this house. "
                        f"Error: {format_exception()}.",
                    )
                    increment_metrics_json(
                        metrics_json_dir=metrics_json_dir,
                        failure=1,
                        worker_ind=worker_ind,
                    )
                    lor_queue.mark_complete(message)
                    update_pbar()
                    continue

            print_error(
                f"[Worker {worker_ind}] processing {traj_id_without_repeat}.",
                flush=True,
            )

            observations_per_repeat = []
            sensor_suite: Optional[SensorSuite] = None
            irrecoverable_failure_in_house = False

            num_sequential_task_sampler_failures = 0
            num_sequential_planner_failures = 0
            retry = 0
            for retry in range(repeats * max_allowed_sequential_planner_failures):
                # Tell the main process that we are still alive
                if is_alive_queue is not None:
                    is_alive_queue.put(True)

                repeat = len(observations_per_repeat)
                if repeat >= repeats:
                    retry -= 1  # We don't want to count this as a retry when we log this counter below
                    break

                if (
                    num_sequential_task_sampler_failures
                    >= max_allowed_sequential_task_sampler_failures
                ):
                    print_error(
                        f"[Worker {worker_ind}] encountered an error when calling next calling next_task for"
                        f" {traj_id_without_repeat} {num_sequential_task_sampler_failures} consecutive times."
                        f" This is unrecoverable, NO ADDITIONAL DATA WILL BE GENERATED FOR THIS HOUSE."
                        f" Traceback:\n {format_exception()}"
                    )
                    irrecoverable_failure_in_house = True
                    num_sequential_irrecoverable_failures += 1
                    retry -= 1  # We don't want to count this as a retry when we log this counter below
                    break

                if (
                    num_sequential_planner_failures
                    >= max_allowed_sequential_planner_failures
                ):
                    print(
                        f"[Worker {worker_ind}] Path planner failed for {traj_id_without_repeat} on repeat {repeat} across"
                        f" {num_sequential_planner_failures} retries."
                        f" This is irrecoverable, NO ADDITIONAL DATA WILL BE GENERATED FOR THIS HOUSE.",
                        flush=True,
                    )
                    irrecoverable_failure_in_house = True
                    num_sequential_irrecoverable_failures += 1
                    retry -= 1  # We don't want to count this as a retry when we log this counter below
                    break

                if task_sampler is None:
                    task_sampler = task_sampler_type(**task_sampler_kwargs)

                traj_id = f"task__{task_sampler.task_type_str.lower()}__{traj_id_without_repeat}_repeat_{repeat:02d}_retry_{retry:02d}"

                seed = (md5_hash_str_as_int(traj_id)) % (2**30)
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)

                task_sampler.set_seed(seed)
                try:
                    task = task_sampler.next_task(
                        force_advance_scene=True, house_index=house_index
                    )
                    num_sequential_task_sampler_failures = 0
                except HouseInvalidForTaskException:
                    # This house cannot generate any tasks because it simply doesn't work for the task sampler
                    # this means we should skip it entirely.
                    print_error(
                        f"[Worker {worker_ind}] encountered a HouseInvalidForTaskException when calling next_task for"
                        f" {traj_id} {num_sequential_task_sampler_failures}. This is unrecoverable,"
                        f" NO DATA WILL BE GENERATED FOR THIS HOUSE. Traceback: {format_exception()}"
                    )
                    irrecoverable_failure_in_house = True
                    # We DO NOT do num_sequential_irrecoverable_failures += 1 here because we don't want to
                    # count this as an irrecoverable failure for the house. These types of exceptions can just happen.
                    break
                except TaskSamplerInInvalidStateError:
                    raise
                except Exception:
                    print_error(
                        f"[Worker {worker_ind}] encountered an error when calling next_task for"
                        f" {traj_id} {num_sequential_task_sampler_failures} consecutive times."
                        f" Traceback: {format_exception()}"
                    )
                    num_sequential_task_sampler_failures += 1
                    continue

                house = task_sampler.current_house

                if sensor_suite is None:
                    sensor_suite = task.sensor_suite

                print(
                    f"[Worker {worker_ind}] starting {traj_id} ({task_sampler.task_type_str}) with instruction: {task.to_string()}",
                    flush=True,
                )
                try:
                    if path_planner.is_planner_guaranteed_to_fail(task):
                        raise PlannerFailedException("Planner is guaranteed to fail.")

                    observations_list = path_planner.plan(task)
                    num_sequential_planner_failures = 0
                    print(
                        f"[Worker {worker_ind}] completed planning for house {house_index} and repeat {repeat}.",
                        flush=True,
                    )
                    increment_metrics_json(
                        metrics_json_dir=metrics_json_dir,
                        planner_successes=1,
                        worker_ind=worker_ind,
                    )
                except IrrecoverablePlannerFailureDueToHouseException:
                    # Similarly as for a HouseInvalidForTaskException, the planner does not think this
                    # house is suitable for this task. We should skip it entirely.
                    print_error(
                        f"[Worker {worker_ind}] encountered a IrrecoverablePlannerFailureDueToHouseException when calling plan for"
                        f" {traj_id} {num_sequential_task_sampler_failures}. This is unrecoverable,"
                        f" NO ADDITIONAL DATA WILL BE GENERATED FOR THIS HOUSE. Traceback: {format_exception()}"
                    )
                    irrecoverable_failure_in_house = True
                    # We DO NOT do num_sequential_irrecoverable_failures += 1 here because we don't want to
                    # count this towards needing the kill this worker. These types of exceptions can just happen.
                    break
                except (PlannerFailedException, AssertionError) as e:
                    # Assertion error catching added for the case where the task hit `max_steps`.
                    if (
                        isinstance(e, AssertionError)
                        and "assert not self.is_done()" not in traceback.format_exc()
                    ):
                        raise

                    if not isinstance(e, TaskDifficultyIncorrectException):
                        increment_metrics_json(
                            metrics_json_dir=metrics_json_dir,
                            planner_failures=1,
                            worker_ind=worker_ind,
                        )

                    num_sequential_planner_failures += 1

                    print_error(
                        f"[Worker {worker_ind}] Failed {traj_id} planning, retrying... Error: {format_exception()}"
                    )
                    continue
                except TimeoutError:
                    print_error(
                        f"[Worker {worker_ind}] TimeoutError for {traj_id}. This suggests that AI2-THOR has died."
                        f" Will mark this trajectory as irrecoverable, attempt to kill any existing AI2-THOR process"
                        f" and restart AI2-THOR."
                    )
                    try:
                        task_sampler.controller.stop()
                    except Exception:
                        pass

                    task_sampler = None  # Should cause the task sampler to be restarted on the next iteration
                    irrecoverable_failure_in_house = True
                    break

                # Normalize task text across the observations in this trajectory.
                for obs in observations_list:
                    if "templated_task_spec" in obs:
                        obs["templated_task_spec"] = observations_list[-1][
                            "templated_task_spec"
                        ]

                observations_dict = batch_observations(
                    observations_list, device=(device if device != -1 else None)
                )
                if device != -1:
                    # Otherwise can lead to CUDA OOM errors
                    to_device_recursively(observations_dict, device=torch.device("cpu"))
                observations_per_repeat.append(observations_dict)

                # Successful generation resets the consecutive-failure count.
                num_sequential_irrecoverable_failures = 0

            if irrecoverable_failure_in_house:
                print(
                    f"[Worker {worker_ind}] irrecoverable failure for house {house_index}.",
                    flush=True,
                )

                if (
                    num_sequential_irrecoverable_failures
                    >= max_allowed_sequential_irrecoverable_failures
                ):
                    # Can only happen if everything fails multiple times in a row, this suggests that something is
                    # really wrong so we quit the worker
                    print_error(
                        f"[Worker {worker_ind}] too many sequential task sampler failures, exiting..."
                    )
                    save_logging_info_for_house(
                        num_saved_trajectories=0,
                        retries=retry + 1,
                    )
                    increment_metrics_json(
                        metrics_json_dir=metrics_json_dir,
                        failure=1,
                        crashed=1,
                        worker_ind=worker_ind,
                    )
                    # This branch exits before the save block's ``finally`` below,
                    # so it must acknowledge the message here. All other branches
                    # are acknowledged by that single ``finally`` block.
                    lor_queue.mark_complete(message)
                    update_pbar()
                    sys.exit(1)

            try:
                if len(observations_per_repeat) != repeats:
                    print_error(
                        f"[Worker {worker_ind}] Generated "
                        f"{len(observations_per_repeat)}/{repeats} trajectories "
                        f"for house {house_index}; refusing to mark it successful."
                    )
                    save_logging_info_for_house(
                        num_saved_trajectories=0,
                        retries=retry + 1,
                    )
                    increment_metrics_json(
                        metrics_json_dir=metrics_json_dir,
                        failure=1,
                        worker_ind=worker_ind,
                    )
                    continue

                if os.path.exists(cur_save_dir):
                    if (
                        len(glob.glob(os.path.join(cur_save_dir, "*.mp4"))) == 0
                        and len(glob.glob(os.path.join(cur_save_dir, "*.hdf5"))) == 0
                    ):
                        print_error(
                            f"[Worker {worker_ind}] Found existing files in the save directory {cur_save_dir}, this should not happen!"
                        )
                        increment_metrics_json(
                            metrics_json_dir=metrics_json_dir,
                            failure=1,
                            worker_ind=worker_ind,
                        )
                        continue

                save_trajectories(
                    observations_list=observations_per_repeat,
                    sensor_suite=sensor_suite,
                    save_dir=cur_save_dir,
                    save_file_suffix="",
                )

                path = os.path.join(cur_save_dir, "house_spec.json")
                with open(path, "w") as f:
                    json.dump(house, f)

                Path(success_save_path).touch()
                save_logging_info_for_house(
                    num_saved_trajectories=len(observations_per_repeat),
                    retries=retry + 1,
                )
                increment_metrics_json(
                    metrics_json_dir=metrics_json_dir, success=1, worker_ind=worker_ind
                )
            except Exception:
                increment_metrics_json(
                    metrics_json_dir=metrics_json_dir, failure=1, worker_ind=worker_ind
                )
                print_error(
                    f"[Worker {worker_ind}] Failed to save trajectories for house {house_index}."
                    f" Error: {format_exception()}"
                )

                try:
                    _remove_incomplete_save_dir(cur_save_dir)
                except Exception:
                    print_error(
                        f"[Worker {worker_ind}] Failed to clean incomplete output "
                        f"at {cur_save_dir}: {format_exception()}"
                    )
                raise
            finally:
                lor_queue.mark_complete(message)
                update_pbar()

            print(
                f"[Worker {worker_ind}] completed house {house_index} with {repeats} repeats.",
                flush=True,
            )

    except queue.Empty:
        print(f"[Worker {worker_ind}] finished.", flush=True)
    except Exception:
        print_error(
            f"[Worker {worker_ind}] encountered an exception."
            f" Last message id {message.message_id if message is not None else None}."
            f" Exception: \n{traceback.format_exc()}"
        )
        raise
    finally:
        print(f"[Worker {worker_ind}] exiting...", flush=True)
        try:
            task_sampler.close()
        except Exception:
            pass

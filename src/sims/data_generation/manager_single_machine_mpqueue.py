import queue
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Sequence, Set, Tuple, Type, Union

import torch
import torch.multiprocessing as mp
from omegaconf import OmegaConf
from setproctitle import setproctitle as ptitle

from allenact.base_abstractions.task import TaskSampler
from allenact.utils.system import get_logger, init_logging
from sims.data_generation.datagen_utils import (
    split_house_repeats_to_id,
    write_config_file,
)
from sims.data_generation.path_planners import PathPlanner
from sims.data_generation.queues import FromToQueue
from sims.data_generation.worker import offline_dataset_worker

mp = (
    mp.get_context("forkserver")
    if torch.cuda.is_available()
    else mp.get_context("spawn")
)

RESULT_POLL_SECONDS = 1
WORKER_GRACEFUL_JOIN_SECONDS = 30
WORKER_STOP_JOIN_SECONDS = 5


@dataclass(frozen=True)
class DatasetGenerationSummary:
    expected_house_indices: Tuple[int, ...]
    successful_house_indices: Tuple[int, ...]
    resumed_house_indices: Tuple[int, ...]

    @property
    def expected_houses(self) -> int:
        return len(self.expected_house_indices)

    @property
    def successful_houses(self) -> int:
        return len(self.successful_house_indices)

    @property
    def failed_house_indices(self) -> Tuple[int, ...]:
        return tuple(
            house_index
            for house_index in self.expected_house_indices
            if house_index not in self.successful_house_indices
        )

    @property
    def failed_houses(self) -> int:
        return len(self.failed_house_indices)

    @property
    def resumed_houses(self) -> int:
        return len(self.resumed_house_indices)


def _record_finished_id(
    finished_id: str, expected_ids: Set[str], finished_ids: Set[str]
) -> bool:
    """Record a valid completion exactly once.

    Workers communicate over a multiprocessing queue, so defensive validation here
    prevents a duplicate or malformed completion from ending the manager early.
    """
    if finished_id not in expected_ids:
        get_logger().warning("Ignoring unexpected completion id %r", finished_id)
        return False
    if finished_id in finished_ids:
        get_logger().warning("Ignoring duplicate completion id %r", finished_id)
        return False

    finished_ids.add(finished_id)
    return True


def _successful_house_indices(
    top_level_save_dir: str, split: str, house_indices: Sequence[int]
) -> Set[int]:
    split_dir = Path(top_level_save_dir) / split
    return {
        house_index
        for house_index in house_indices
        if (split_dir / f"{house_index:06d}" / "success.txt").is_file()
    }


def _summarize_dataset_outcome(
    top_level_save_dir: str,
    split: str,
    house_indices: Sequence[int],
    resumed_house_indices: Set[int],
) -> DatasetGenerationSummary:
    expected = tuple(sorted(set(house_indices)))
    successful = tuple(
        sorted(_successful_house_indices(top_level_save_dir, split, expected))
    )
    resumed = tuple(sorted(set(successful) & resumed_house_indices))
    summary = DatasetGenerationSummary(
        expected_house_indices=expected,
        successful_house_indices=successful,
        resumed_house_indices=resumed,
    )

    if summary.expected_houses and summary.successful_houses == 0:
        raise RuntimeError(
            f"Dataset generation produced no successful houses (0/{summary.expected_houses}); "
            f"no success.txt markers were found under "
            f"{Path(top_level_save_dir) / split}"
        )

    if summary.failed_houses:
        get_logger().warning(
            "Dataset generation completed partially: %d/%d houses succeeded; "
            "failed house indices: %s",
            summary.successful_houses,
            summary.expected_houses,
            summary.failed_house_indices,
        )
    else:
        get_logger().info(
            "Dataset generation completed: %d/%d houses succeeded (%d resumed)",
            summary.successful_houses,
            summary.expected_houses,
            summary.resumed_houses,
        )

    return summary


def _check_worker_health(processes, remaining_results: int) -> None:
    failed_processes = [
        process
        for process in processes
        if process.exitcode is not None and process.exitcode != 0
    ]
    if failed_processes:
        raise RuntimeError(
            "Dataset workers exited before returning all results; exit codes: "
            f"{[process.exitcode for process in failed_processes]}"
        )

    if (
        remaining_results
        and processes
        and all(process.exitcode is not None for process in processes)
    ):
        raise RuntimeError(
            f"All dataset workers exited with {remaining_results} result(s) still pending"
        )


def _join_processes_with_deadline(processes, timeout_seconds: float) -> None:
    deadline = time.monotonic() + timeout_seconds
    for process in processes:
        process.join(max(0.0, deadline - time.monotonic()))


def _join_and_verify_workers(processes) -> None:
    _join_processes_with_deadline(processes, WORKER_GRACEFUL_JOIN_SECONDS)

    alive_processes = [process for process in processes if process.is_alive()]
    if alive_processes:
        raise RuntimeError(
            f"{len(alive_processes)} dataset worker(s) did not exit after processing"
        )

    failed_exit_codes = [
        process.exitcode
        for process in processes
        if process.exitcode is not None and process.exitcode != 0
    ]
    if failed_exit_codes:
        raise RuntimeError(
            f"Dataset workers exited unsuccessfully: {failed_exit_codes}"
        )


def _stop_workers(processes) -> None:
    for process in processes:
        if process.is_alive():
            process.terminate()
    _join_processes_with_deadline(processes, WORKER_STOP_JOIN_SECONDS)

    stubborn_processes = [process for process in processes if process.is_alive()]
    for process in stubborn_processes:
        process.kill()
    _join_processes_with_deadline(stubborn_processes, WORKER_STOP_JOIN_SECONDS)

    still_alive = [process for process in stubborn_processes if process.is_alive()]
    if still_alive:
        get_logger().error(
            "%d dataset worker(s) remained alive after terminate and kill",
            len(still_alive),
        )


def manager_single_machine_mpqueue(
    nworkers: int,
    house_repeats: int,
    split: str,
    top_level_save_dir: str,
    constants: OmegaConf,
    task_sampler_type: Type[TaskSampler],
    device_to_task_sampler_kwargs: Callable[[Union[int]], Dict[str, Any]],
    path_planner: PathPlanner,
) -> DatasetGenerationSummary:
    ptitle("Offline Expert Dataset Manager")

    init_logging()

    if nworkers < 1:
        raise ValueError("nworkers must be at least 1")

    to_worker_queue = mp.Queue()
    from_worker_queue = mp.Queue()

    devices = (
        [None]
        if not torch.cuda.is_available()
        else list(range(torch.cuda.device_count()))
    )

    task_sampler_kwargs_per_device = []
    for d in devices:
        task_sampler_kwargs_per_device.append(device_to_task_sampler_kwargs(d))

    # Fail before inspecting success markers or starting workers if this output
    # directory belongs to a generation run with different semantics.
    write_config_file(
        cfg=constants,
        top_level_save_dir=top_level_save_dir,
    )

    house_inds = sorted(
        list(
            set(sum((tsk["house_inds"] for tsk in task_sampler_kwargs_per_device), []))
        )
    )
    resumed_house_inds = _successful_house_indices(
        top_level_save_dir, split, house_inds
    )

    ids = []
    for i in house_inds:
        ids.append(
            split_house_repeats_to_id(split=split, house=i, repeats=house_repeats)
        )
    get_logger().info(f"Generated {len(ids)} job ids for workers to process")
    expected_ids = set(ids)
    ids.reverse()

    # 32767 is the max number of items that can be put in an mp.Queue
    # we'll start by putting that many items in the queue and add more later below
    for _ in range(32767):
        if len(ids) == 0:
            break
        to_worker_queue.put(ids.pop())

    processes = []
    finished_ids = set()
    completed_normally = False
    try:
        for i in range(nworkers):
            p = mp.Process(
                target=offline_dataset_worker,
                kwargs=dict(
                    constants=constants,
                    task_sampler_type=task_sampler_type,
                    task_sampler_kwargs=task_sampler_kwargs_per_device[
                        i % len(devices)
                    ],
                    expected_split=split,
                    worker_ind=i,
                    lor_queue=FromToQueue(
                        from_queue=to_worker_queue,
                        to_queue=from_worker_queue,
                    ),
                    device=devices[i % len(devices)],
                    path_planner=path_planner,
                    top_level_save_dir=top_level_save_dir,
                    metrics_json_dir=str(Path(top_level_save_dir) / "generation_logs"),
                ),
            )
            p.start()
            time.sleep(1)
            processes.append(p)

        last_done = time.time()
        last_ping = time.time()
        while finished_ids != expected_ids:
            try:
                finished_id = from_worker_queue.get(timeout=RESULT_POLL_SECONDS)
                if not _record_finished_id(
                    finished_id, expected_ids=expected_ids, finished_ids=finished_ids
                ):
                    continue

                get_logger().info(f"{finished_id} finished processing.")
                last_done = time.time()
                last_ping = last_done

                if len(ids) > 0:
                    to_worker_queue.put(ids.pop())
            except queue.Empty:
                _check_worker_health(
                    processes, remaining_results=len(expected_ids - finished_ids)
                )
                if time.time() - last_ping > 60:
                    get_logger().info("Manager: no new results in 60 seconds.")
                    last_ping = time.time()

            if time.time() - last_done > 60 * 60 * 1:
                raise TimeoutError

        _join_and_verify_workers(processes)
        summary = _summarize_dataset_outcome(
            top_level_save_dir=top_level_save_dir,
            split=split,
            house_indices=house_inds,
            resumed_house_indices=resumed_house_inds,
        )
        completed_normally = True
        return summary
    finally:
        get_logger().info(
            f"Dataset manager quitting ({len(finished_ids)}/{len(expected_ids)}"
            f" scenes complete) ..."
        )
        if not completed_normally:
            _stop_workers(processes)

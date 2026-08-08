"""Focused regression tests for offline dataset worker reliability."""

import errno
import queue
import shutil
from pathlib import Path
from unittest.mock import patch

import pytest
from omegaconf import OmegaConf

from sims.data_generation.queues import QueueMessage
from sims.data_generation.worker import (
    _remove_incomplete_save_dir,
    offline_dataset_worker,
)
from sims.utils.data_generation_utils.exception_utils import (
    IrrecoverablePlannerFailureDueToHouseException,
)


class _SingleMessageQueue:
    def __init__(self, body):
        self.message = QueueMessage(body)
        self.completed = []

    def get(self, timeout=None):
        if self.message is None:
            raise queue.Empty
        message, self.message = self.message, None
        return message

    def mark_complete(self, message):
        self.completed.append(message.body)


class _Task:
    sensor_suite = object()

    @staticmethod
    def to_string():
        return "explore"


class _TaskSampler:
    task_type_str = "HouseWalkthrough"

    def __init__(self, **kwargs):
        self.current_house = {"rooms": []}

    def set_seed(self, seed):
        self.seed = seed

    def next_task(self, **kwargs):
        return _Task()

    def close(self):
        pass


class _PartialPlanner:
    def __init__(self):
        self.calls = 0

    @staticmethod
    def is_planner_guaranteed_to_fail(task):
        return False

    def plan(self, task):
        self.calls += 1
        if self.calls == 1:
            return [{}]
        raise IrrecoverablePlannerFailureDueToHouseException("invalid house")


class _SuccessfulPlanner:
    @staticmethod
    def is_planner_guaranteed_to_fail(task):
        return False

    @staticmethod
    def plan(task):
        return [{}]


def _run_worker(tmp_path, *, repeats, planner):
    message_body = f"split_val__house_00000000__repeats_{repeats:02d}"
    work_queue = _SingleMessageQueue(message_body)
    offline_dataset_worker(
        constants=OmegaConf.create({}),
        task_sampler_type=_TaskSampler,
        task_sampler_kwargs={"task_args": {"action_space": object()}},
        path_planner=planner,
        worker_ind=0,
        lor_queue=work_queue,
        device=None,
        expected_split="val",
        top_level_save_dir=str(tmp_path),
        metrics_json_dir=None,
    )
    return work_queue


def test_enotempty_cleanup_targets_house_and_removes_files_and_directories(tmp_path):
    house_dir = tmp_path / "val" / "000000"
    nested_dir = house_dir / "nested"
    nested_dir.mkdir(parents=True)
    (house_dir / "trajectory.mp4").write_bytes(b"partial")
    (nested_dir / "annotation.jsonl").write_text("{}\n")

    real_rmtree = shutil.rmtree
    root_calls = 0

    def fail_first_root_removal(path, *args, **kwargs):
        nonlocal root_calls
        if Path(path) == house_dir and root_calls == 0:
            root_calls += 1
            raise OSError(errno.ENOTEMPTY, "directory not empty")
        return real_rmtree(path, *args, **kwargs)

    with patch(
        "sims.data_generation.worker.shutil.rmtree",
        side_effect=fail_first_root_removal,
    ):
        _remove_incomplete_save_dir(str(house_dir))

    assert not house_dir.exists()


@patch("sims.data_generation.worker.batch_observations", return_value={"obs": []})
@patch("sims.data_generation.worker.write_config_file")
@patch("sims.data_generation.worker.save_trajectories")
def test_partial_repeat_set_is_not_saved_or_marked_successful(
    mock_save_trajectories, mock_write_config, mock_batch, tmp_path
):
    work_queue = _run_worker(tmp_path, repeats=2, planner=_PartialPlanner())

    mock_save_trajectories.assert_not_called()
    assert not (tmp_path / "val" / "000000" / "success.txt").exists()
    assert work_queue.completed == ["split_val__house_00000000__repeats_02"]


@patch("sims.data_generation.worker.batch_observations", return_value={"obs": []})
@patch("sims.data_generation.worker.write_config_file")
def test_complete_repeat_set_is_saved_before_success_marker(
    mock_write_config, mock_batch, tmp_path
):
    house_dir = tmp_path / "val" / "000000"

    def save_complete_set(*args, **kwargs):
        Path(kwargs["save_dir"]).mkdir(parents=True)

    with patch(
        "sims.data_generation.worker.save_trajectories",
        side_effect=save_complete_set,
    ) as mock_save:
        _run_worker(tmp_path, repeats=2, planner=_SuccessfulPlanner())

    assert len(mock_save.call_args.kwargs["observations_list"]) == 2
    assert (house_dir / "success.txt").is_file()


@patch("sims.data_generation.worker.batch_observations", return_value={"obs": []})
@patch("sims.data_generation.worker.write_config_file")
def test_save_failure_removes_partial_tree_and_propagates(
    mock_write_config, mock_batch, tmp_path
):
    house_dir = tmp_path / "val" / "000000"

    def fail_after_partial_save(*args, **kwargs):
        nested_dir = house_dir / "nested"
        nested_dir.mkdir(parents=True)
        (house_dir / "partial.mp4").write_bytes(b"partial")
        (nested_dir / "partial.jsonl").write_text("{}\n")
        raise RuntimeError("encoder failed")

    with patch(
        "sims.data_generation.worker.save_trajectories",
        side_effect=fail_after_partial_save,
    ):
        with pytest.raises(RuntimeError, match="encoder failed"):
            _run_worker(tmp_path, repeats=1, planner=_SuccessfulPlanner())

    assert not house_dir.exists()
    assert not (house_dir / "success.txt").exists()

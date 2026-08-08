"""Regression tests for multiprocessing manager bookkeeping."""

import queue
import json
from unittest.mock import patch

import pytest
from omegaconf import OmegaConf

from sims.data_generation.manager_single_machine_mpqueue import (
    _check_worker_health,
    _join_and_verify_workers,
    _record_finished_id,
    _stop_workers,
    _summarize_dataset_outcome,
    manager_single_machine_mpqueue,
)
from sims.qa.spatial_metadata_gen_mp import (
    _collect_worker_results,
    _discover_scene_paths,
    _summarize_worker_results,
)
from sims.qa.spatial_metadata_gen import process_house


class _FakeProcess:
    def __init__(self, *, exitcode=None, alive=True):
        self.exitcode = exitcode
        self._alive = alive
        self.join_calls = 0
        self.terminate_calls = 0
        self.kill_calls = 0

    def is_alive(self):
        return self._alive

    def join(self, timeout=None):
        self.join_calls += 1

    def terminate(self):
        self.terminate_calls += 1
        self._alive = False
        self.exitcode = -15

    def kill(self):
        self.kill_calls += 1
        self._alive = False
        self.exitcode = -9


class _StubbornProcess(_FakeProcess):
    def terminate(self):
        self.terminate_calls += 1


class _CaptureProcess(_FakeProcess):
    instances = []

    def __init__(self, *, target, kwargs):
        super().__init__(exitcode=0, alive=False)
        self.target = target
        self.kwargs = kwargs
        self.__class__.instances.append(self)

    def start(self):
        pass


def test_duplicate_completion_does_not_count_twice():
    expected_ids = {"house-1", "house-2"}
    finished_ids = set()

    assert _record_finished_id("house-1", expected_ids, finished_ids) is True
    assert _record_finished_id("house-1", expected_ids, finished_ids) is False
    assert finished_ids == {"house-1"}

    assert _record_finished_id("house-2", expected_ids, finished_ids) is True
    assert finished_ids == expected_ids


def test_dataset_worker_failure_is_reported_immediately():
    with pytest.raises(RuntimeError, match=r"exit codes: \[3\]"):
        _check_worker_health(
            [_FakeProcess(exitcode=3, alive=False)], remaining_results=1
        )


def test_dataset_manager_rejects_workers_exiting_before_all_results():
    with pytest.raises(RuntimeError, match="All dataset workers exited"):
        _check_worker_health(
            [_FakeProcess(exitcode=0, alive=False)], remaining_results=2
        )


def test_dataset_worker_exit_codes_are_verified_after_join():
    process = _FakeProcess(exitcode=7, alive=False)

    with pytest.raises(RuntimeError, match=r"unsuccessfully: \[7\]"):
        _join_and_verify_workers([process])

    assert process.join_calls == 1


def test_dataset_worker_cleanup_terminates_and_joins_survivors():
    running = _FakeProcess(exitcode=None, alive=True)
    finished = _FakeProcess(exitcode=0, alive=False)

    _stop_workers([running, finished])

    assert running.terminate_calls == 1
    assert finished.terminate_calls == 0
    assert running.join_calls == 1
    assert finished.join_calls == 1


def test_dataset_worker_cleanup_kills_survivors_that_ignore_terminate():
    process = _StubbornProcess(exitcode=None, alive=True)

    _stop_workers([process])

    assert process.terminate_calls == 1
    assert process.kill_calls == 1
    assert process.join_calls == 2
    assert process.is_alive() is False


@patch("sims.data_generation.manager_single_machine_mpqueue.init_logging")
@patch("sims.data_generation.manager_single_machine_mpqueue.write_config_file")
def test_dataset_config_is_validated_before_resume_markers(
    mock_write_config, mock_init_logging, tmp_path
):
    mock_write_config.side_effect = ValueError("configuration mismatch")

    with patch(
        "sims.data_generation.manager_single_machine_mpqueue._successful_house_indices"
    ) as mock_successful:
        with pytest.raises(ValueError, match="configuration mismatch"):
            manager_single_machine_mpqueue(
                nworkers=1,
                house_repeats=2,
                split="val",
                top_level_save_dir=str(tmp_path),
                constants=OmegaConf.create({"task": "HouseWalkthrough"}),
                task_sampler_type=object,
                device_to_task_sampler_kwargs=lambda device: {
                    "house_inds": [0],
                    "task_args": {"action_space": object()},
                },
                path_planner=object(),
            )

    mock_successful.assert_not_called()


@patch("sims.data_generation.manager_single_machine_mpqueue.time.sleep")
@patch("sims.data_generation.manager_single_machine_mpqueue.init_logging")
@patch("sims.data_generation.manager_single_machine_mpqueue.write_config_file")
def test_dataset_worker_logs_are_scoped_to_dataset_output(
    mock_write_config, mock_init_logging, mock_sleep, tmp_path
):
    _CaptureProcess.instances.clear()
    with patch(
        "sims.data_generation.manager_single_machine_mpqueue.mp.Process",
        _CaptureProcess,
    ):
        manager_single_machine_mpqueue(
            nworkers=1,
            house_repeats=2,
            split="val",
            top_level_save_dir=str(tmp_path),
            constants=OmegaConf.create({"task": "HouseWalkthrough"}),
            task_sampler_type=object,
            device_to_task_sampler_kwargs=lambda device: {
                "house_inds": [],
                "task_args": {"action_space": object()},
            },
            path_planner=object(),
        )

    assert _CaptureProcess.instances[0].kwargs["metrics_json_dir"] == str(
        tmp_path / "generation_logs"
    )


def _touch_success_marker(output_dir, split, house_index):
    marker = output_dir / split / f"{house_index:06d}" / "success.txt"
    marker.parent.mkdir(parents=True)
    marker.touch()


def test_dataset_outcome_raises_when_every_house_failed(tmp_path):
    with pytest.raises(RuntimeError, match=r"no successful houses \(0/2\)"):
        _summarize_dataset_outcome(
            top_level_save_dir=str(tmp_path),
            split="val",
            house_indices=[0, 1],
            resumed_house_indices=set(),
        )


@patch("sims.data_generation.manager_single_machine_mpqueue.get_logger")
def test_partial_dataset_outcome_warns_and_counts_resumed_markers(
    mock_get_logger, tmp_path
):
    _touch_success_marker(tmp_path, "val", 1)

    summary = _summarize_dataset_outcome(
        top_level_save_dir=str(tmp_path),
        split="val",
        house_indices=[0, 1],
        resumed_house_indices={1},
    )

    assert summary.expected_houses == 2
    assert summary.successful_houses == 1
    assert summary.failed_house_indices == (0,)
    assert summary.failed_houses == 1
    assert summary.resumed_house_indices == (1,)
    assert summary.resumed_houses == 1
    mock_get_logger.return_value.warning.assert_called_once()


def test_dataset_outcome_counts_new_and_resumed_successes(tmp_path):
    _touch_success_marker(tmp_path, "val", 0)
    _touch_success_marker(tmp_path, "val", 1)

    summary = _summarize_dataset_outcome(
        top_level_save_dir=str(tmp_path),
        split="val",
        house_indices=[0, 1],
        resumed_house_indices={0},
    )

    assert summary.successful_house_indices == (0, 1)
    assert summary.failed_house_indices == ()
    assert summary.resumed_house_indices == (0,)


def test_discover_scene_paths_excludes_split_root_files(tmp_path):
    first = tmp_path / "000001"
    second = tmp_path / "000002"
    first.mkdir()
    second.mkdir()
    (tmp_path / "combined_qa_pairs.jsonl").write_text("{}\n")

    assert _discover_scene_paths(str(tmp_path)) == [str(first), str(second)]


def test_collects_one_metadata_result_per_scene():
    results_queue = queue.Queue()
    results_queue.put(("000001", "success"))
    results_queue.put(("000002", "success"))

    assert _collect_worker_results(
        results_queue, [_FakeProcess()], expected_count=2
    ) == [
        ("000001", "success"),
        ("000002", "success"),
    ]


def test_metadata_collection_fails_when_worker_dies():
    with pytest.raises(RuntimeError, match="exited before returning"):
        _collect_worker_results(
            queue.Queue(), [_FakeProcess(exitcode=3, alive=False)], expected_count=1
        )


def test_metadata_summary_rejects_partial_output_by_default(tmp_path):
    results = [("000001", "success"), ("000002", "failed: invalid scene")]

    with pytest.raises(RuntimeError, match="1 scene.*failed metadata"):
        _summarize_worker_results(results, tmp_path)


def test_metadata_summary_allows_explicit_partial_output(tmp_path):
    results = [("000001", "success"), ("000002", "failed: invalid scene")]

    assert _summarize_worker_results(results, tmp_path, allow_partial=True) == {
        "success": 1,
        "failed": 1,
    }


def test_failed_metadata_overwrite_removes_stale_output(tmp_path):
    scene_dir = tmp_path / "000001"
    scene_dir.mkdir()
    house_spec = scene_dir / "house_spec.json"
    house_spec.write_text(json.dumps({"rooms": [], "objects": []}))
    metadata = scene_dir / "spatial_metadata.json"
    metadata.write_text('{"stale": true}\n')

    class FailingController:
        def reset(self, scene):
            raise RuntimeError("simulator failed")

    with pytest.raises(RuntimeError, match="simulator failed"):
        process_house(
            FailingController(),
            str(house_spec),
            save=True,
            overwrite=True,
        )

    assert not metadata.exists()

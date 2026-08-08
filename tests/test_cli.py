from unittest.mock import patch

import pytest

from sims.cli import main
from sims.data_generation.arg_parsers import get_arg_parser_for_offline_datagen
from sims.data_generation.paths import (
    configured_objaverse_data_dir,
    resolve_objaverse_data_dir,
)


def test_root_help_is_lightweight(capsys):
    assert main([]) == 0

    help_text = capsys.readouterr().out
    assert "generate" in help_text
    assert "qa" in help_text


@pytest.mark.parametrize("command", ["generate", "qa"])
def test_subcommand_help(command, capsys):
    with pytest.raises(SystemExit) as error:
        main([command, "--help"])

    assert error.value.code == 0
    help_text = capsys.readouterr().out
    assert "--task-type" not in help_text
    assert "--action-space" not in help_text


def test_generate_dispatches_without_reparsing():
    with patch(
        "sims.data_generation.datagen_scripts.offline_video_datagen_mpqueue.main",
        return_value=7,
    ) as generate:
        assert main(["generate", "--dataset-dir", "output/demo"]) == 7

    generate.assert_called_once_with(
        ["--dataset-dir", "output/demo"], prog="sims-v generate"
    )


def test_qa_dispatches_without_reparsing():
    with patch("sims.pipeline.main", return_value=9) as qa:
        assert main(["qa", "--dataset-dir", "output/demo"]) == 9

    qa.assert_called_once_with(["--dataset-dir", "output/demo"], prog="sims-v qa")


@pytest.mark.parametrize(
    "modality_args, expected_extra_sensors, expected_modalities, render_depth, render_class",
    [
        ([], ("offline_annos",), ("rgb",), False, False),
        (
            ["--extra-video-modalities", "edge", "semantic_seg", "depth"],
            ("offline_annos", "depth", "semantic_seg", "edge"),
            ("rgb", "depth", "semantic_seg", "edge"),
            True,
            True,
        ),
    ],
)
def test_generation_runner_wires_the_public_walkthrough_pipeline(
    modality_args,
    expected_extra_sensors,
    expected_modalities,
    render_depth,
    render_class,
):
    from sims.data_generation.datagen_scripts.offline_video_datagen_mpqueue import run
    from sims.environment.action_spaces import DiscreteStretchActionSpace
    from sims.tasks.house_walkthrough_task import HouseWalkthroughTask
    from sims.tasks.house_walkthrough_task_sampler import HouseWalkthroughTaskSampler

    class Dataset:
        def __init__(self):
            self.items = [object(), object()]

        def __len__(self):
            return len(self.items)

        def select(self, indices):
            self.items = [self.items[index] for index in indices]
            return self

    args = get_arg_parser_for_offline_datagen().parse_args(
        [
            "--dataset-dir",
            "output/demo",
            "--max-houses",
            "1",
            "--workers",
            "2",
            "--trajectories-per-house",
            "2",
            "--rotation-noise-std-degrees",
            "0.5",
            "--resolution-scale",
            "1.5",
            *modality_args,
        ]
    )
    dataset = Dataset()
    task_args = {"task": "arguments"}
    generation_config = {"schema_version": 4}

    with (
        patch("torch.cuda.device_count", return_value=1),
        patch(
            "sims.data_generation.datagen_scripts.task_datagen_utils.default_workers_per_device",
            return_value=3,
        ),
        patch(
            "sims.data_generation.datagen_scripts.task_datagen_utils.get_house_dataset",
            return_value={"val": dataset},
        ),
        patch(
            "sims.data_generation.datagen_scripts.task_datagen_utils.get_walkthrough_task_args",
            return_value=task_args,
        ) as build_task_args,
        patch(
            "sims.data_generation.datagen_utils.build_generation_config",
            return_value=generation_config,
        ) as build_config,
        patch(
            "sims.data_generation.manager_single_machine_mpqueue.manager_single_machine_mpqueue"
        ) as manager,
    ):
        run(args)

    action_space = build_task_args.call_args.kwargs["action_space"]
    assert isinstance(action_space, DiscreteStretchActionSpace)
    assert build_task_args.call_args.kwargs["width"] == 594
    assert build_task_args.call_args.kwargs["height"] == 336
    assert (
        tuple(
            sensor.uuid for sensor in build_task_args.call_args.kwargs["extra_sensors"]
        )
        == expected_extra_sensors
    )
    assert build_config.call_args.kwargs["task_type"] == "HouseWalkthrough"
    assert build_config.call_args.kwargs["video_modalities"] == expected_modalities

    manager_args = manager.call_args.kwargs
    assert manager_args["nworkers"] == 2
    assert manager_args["house_repeats"] == 2
    assert manager_args["top_level_save_dir"] == "output/demo"
    assert manager_args["constants"] is generation_config
    assert manager_args["task_sampler_type"] is HouseWalkthroughTaskSampler

    sampler_args = manager_args["device_to_task_sampler_kwargs"](0)
    assert sampler_args["task_args"] is task_args
    assert sampler_args["task_type"] is HouseWalkthroughTask
    assert sampler_args["sample_per_house"] == 2
    assert sampler_args["controller_args"]["rotation_noise_std_degrees"] == 0.5
    assert sampler_args["controller_args"]["width"] == 594
    assert sampler_args["controller_args"]["height"] == 336
    assert sampler_args["controller_args"]["renderDepthImage"] is render_depth
    assert sampler_args["controller_args"]["renderSemanticSegmentation"] is render_class
    assert sampler_args["controller_args"]["renderInstanceSegmentation"] is True


def test_walkthrough_sensor_spaces_follow_render_resolution():
    from sims.data_generation.datagen_scripts.task_datagen_utils import (
        get_walkthrough_task_args,
    )
    from sims.environment.action_spaces import DiscreteStretchActionSpace

    task_args = get_walkthrough_task_args(
        max_steps=10,
        action_space=DiscreteStretchActionSpace(),
        include_manipulation_sensor=True,
        width=680,
        height=384,
    )
    sensor_spaces = {
        sensor.uuid: sensor.observation_space.shape for sensor in task_args["sensors"]
    }

    assert sensor_spaces["rgb"] == (384, 668, 3)
    assert sensor_spaces["raw_manipulation_camera"] == (384, 668, 3)
    assert "raw_navigation_depth" not in sensor_spaces
    assert "raw_manipulation_depth" not in sensor_spaces


def test_objaverse_path_resolution_precedence(tmp_path):
    from_environment = tmp_path / "from-environment"
    explicit = tmp_path / "explicit"
    for root in (from_environment, explicit):
        for subdirectory in ("processed", "houses", "procthor_databases"):
            (root / subdirectory).mkdir(parents=True)

    environ = {"OBJAVERSE_DATA_DIR": str(from_environment)}
    resolved = resolve_objaverse_data_dir(
        explicit,
        required=True,
        cwd=tmp_path,
        environ=environ,
    )

    assert resolved == explicit.resolve()
    assert environ["OBJAVERSE_DATA_DIR"] == str(explicit.resolve())


def test_objaverse_path_auto_detects_conventional_directory(tmp_path):
    root = tmp_path / "objaverse_sims"
    root.mkdir()
    environ = {}

    assert resolve_objaverse_data_dir(cwd=tmp_path, environ=environ) == root


def test_configured_objaverse_path_is_read_at_call_time(tmp_path):
    first = configured_objaverse_data_dir(
        cwd=tmp_path,
        environ={"OBJAVERSE_DATA_DIR": str(tmp_path / "first")},
    )
    second = configured_objaverse_data_dir(
        cwd=tmp_path,
        environ={"OBJAVERSE_DATA_DIR": str(tmp_path / "second")},
    )

    assert first == (tmp_path / "first").resolve()
    assert second == (tmp_path / "second").resolve()


def test_controller_hook_uses_current_objaverse_root(tmp_path, monkeypatch):
    from sims.utils.constants.stretch_initialization_utils import get_stretch_env_args

    monkeypatch.setenv("OBJAVERSE_DATA_DIR", str(tmp_path))
    hook = get_stretch_env_args()["action_hook_runner"]

    assert hook.asset_directory == str((tmp_path / "processed").resolve())


def test_missing_required_objaverse_data_has_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="--objaverse-dir"):
        resolve_objaverse_data_dir(required=True, cwd=tmp_path, environ={})


def test_generate_cli_reports_missing_objaverse_assets(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OBJAVERSE_DATA_DIR", raising=False)

    with pytest.raises(SystemExit) as error:
        main(
            [
                "generate",
                "--dataset-dir",
                "output/demo",
                "--house-dataset",
                "objaverse",
            ]
        )

    assert error.value.code == 2
    assert "Objaverse assets were not found" in capsys.readouterr().err


def test_explicit_objaverse_path_must_exist(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        resolve_objaverse_data_dir(tmp_path / "missing", environ={})


def test_worker_default_is_pure_and_bounded():
    from sims.data_generation.datagen_scripts.task_datagen_utils import (
        default_workers_per_device,
    )

    assert default_workers_per_device({}, platform_name="darwin") == 1
    assert default_workers_per_device({}, platform_name="linux") == 3
    assert (
        default_workers_per_device(
            {"ASSIGNED_MEMORY_BYTES": "1"}, platform_name="linux"
        )
        == 1
    )

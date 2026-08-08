"""Tests for the public walkthrough-generation arguments."""

import pytest

from sims.data_generation.arg_parsers import (
    EXTRA_VIDEO_MODALITIES,
    get_arg_parser_for_offline_datagen,
    resolve_extra_video_modalities,
)


def parse(*args):
    return get_arg_parser_for_offline_datagen().parse_args(
        ["--dataset-dir", "out/demo", *args]
    )


def test_generation_defaults_are_small_and_disable_added_randomization():
    args = parse()

    assert args.dataset_dir == "out/demo"
    assert args.split == "val"
    assert args.house_dataset == "procthor"
    assert args.trajectories_per_house == 1
    assert args.max_houses == 1
    assert args.material_randomization_probability == 0
    assert args.rotation_noise_std_degrees == 0
    assert args.workers is None
    assert args.extra_video_modalities == ()


@pytest.mark.parametrize("house_dataset", ["procthor", "objaverse"])
def test_supported_house_datasets(house_dataset):
    assert parse("--house-dataset", house_dataset).house_dataset == house_dataset


def test_unknown_house_dataset_is_rejected():
    with pytest.raises(SystemExit):
        parse("--house-dataset", "unsupported")


def test_rendering_and_randomization_options():
    args = parse(
        "--resolution-scale",
        "1.5",
        "--material-randomization-probability",
        "0.25",
        "--rotation-noise-std-degrees",
        "0.5",
        "--trajectories-per-house",
        "2",
        "--workers",
        "3",
    )

    assert args.resolution_scale == 1.5
    assert args.material_randomization_probability == 0.25
    assert args.rotation_noise_std_degrees == 0.5
    assert args.trajectories_per_house == 2
    assert args.workers == 3


def test_extra_video_modalities_are_opt_in_and_normalized():
    args = parse(
        "--extra-video-modalities",
        "semantic_seg",
        "depth",
        "semantic_seg",
    )

    assert resolve_extra_video_modalities(args.extra_video_modalities) == (
        "depth",
        "semantic_seg",
    )
    assert resolve_extra_video_modalities(("all",)) == EXTRA_VIDEO_MODALITIES


def test_unknown_extra_video_modality_is_rejected():
    with pytest.raises(SystemExit):
        parse("--extra-video-modalities", "normals")


@pytest.mark.parametrize(
    "args",
    [
        ("--max-houses", "0"),
        ("--trajectories-per-house", "0"),
        ("--resolution-scale", "0"),
        ("--material-randomization-probability", "1.1"),
        ("--rotation-noise-std-degrees", "-0.1"),
        ("--rotation-noise-std-degrees", "nan"),
        ("--rotation-noise-std-degrees", "inf"),
        ("--resolution-scale", "inf"),
    ],
)
def test_invalid_numeric_values_are_rejected(args):
    with pytest.raises(SystemExit):
        parse(*args)


def test_help_omits_internal_task_and_action_names():
    help_text = get_arg_parser_for_offline_datagen().format_help()

    assert "task-type" not in help_text
    assert "action-space" not in help_text
    assert "--debug" not in help_text
    assert "--extra-video-modalities" in help_text
    assert "default: none" in help_text

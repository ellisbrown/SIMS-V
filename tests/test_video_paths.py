"""Tests for canonical and released RGB video filenames."""

import pytest

from sims.video_paths import (
    index_rgb_video_filenames,
    qa_seed_video_identity,
    video_path_for_version,
)


def test_index_rgb_video_filenames_accepts_canonical_and_legacy_names():
    assert index_rgb_video_filenames(["rgb__0.mp4"]) == {"0": "rgb__0.mp4"}
    assert index_rgb_video_filenames(["raw_navigation_camera__0.mp4"]) == {
        "0": "raw_navigation_camera__0.mp4"
    }
    assert index_rgb_video_filenames(
        ["rgb__0.mp4", "raw_navigation_camera__1.mp4", "depth__0.mp4"]
    ) == {
        "0": "rgb__0.mp4",
        "1": "raw_navigation_camera__1.mp4",
    }


def test_index_rgb_video_filenames_rejects_two_names_for_one_trajectory():
    with pytest.raises(ValueError, match="multiple RGB videos for trajectory '0'"):
        index_rgb_video_filenames(["rgb__0.mp4", "raw_navigation_camera__0.mp4"])


@pytest.mark.parametrize(
    ("filename", "video_version", "expected"),
    [
        ("val/scene/rgb__0.mp4", "rgb", "val/scene/rgb__0.mp4"),
        (
            "val/scene/raw_navigation_camera__0.mp4",
            "rgb",
            "val/scene/raw_navigation_camera__0.mp4",
        ),
        ("val/scene/rgb__0.mp4", "depth", "val/scene/depth__0.mp4"),
        (
            "val/scene/raw_navigation_camera__0.mp4",
            "depth",
            "val/scene/depth__0.mp4",
        ),
        (
            "datasets/rgb-release/scene/rgb__3.mp4",
            "semantic_seg",
            "datasets/rgb-release/scene/semantic_seg__3.mp4",
        ),
    ],
)
def test_video_path_for_version(filename, video_version, expected):
    assert video_path_for_version(filename, video_version) == expected


def test_video_path_for_version_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="Invalid video_version"):
        video_path_for_version("scene/rgb__0.mp4", "wireframe")
    with pytest.raises(ValueError, match="Not a recognized RGB video path"):
        video_path_for_version("scene/video.mp4", "rgb")


def test_rgb_video_path_preserves_the_exact_input_spelling():
    filename = r"val\scene\raw_navigation_camera__0.mp4"
    assert video_path_for_version(filename, "rgb") == filename


def test_rgb_rename_preserves_qa_seed_identity():
    assert qa_seed_video_identity("val/scene/rgb__0.mp4") == qa_seed_video_identity(
        "val/scene/raw_navigation_camera__0.mp4"
    )

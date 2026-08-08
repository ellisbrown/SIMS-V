"""Tests for QA format stage video path construction."""

import json

import pytest


from sims.pipeline import _is_format_complete
from sims.qa.qa_to_ov_train_multiturn import (
    build_multi_turn_convos,
    create_multiturn_jsonl,
    group_qas_for_multiturn,
)


def _make_qa_entry(filename, source="my_dataset", task="obj_count", idx=0):
    """Helper to create a minimal QA entry matching the combined JSONL format."""
    return {
        "idx": idx,
        "type": "v0",
        "filename": filename,
        "source": source,
        "task": task,
        "question": "How many chairs?",
        "gt_answer": "2",
        "mc_question": "How many chairs?\nA. 1\nB. 2\nC. 3\nD. 4",
        "mc_answer": "B",
        "mc_choices": ["A. 1", "B. 2", "C. 3", "D. 4"],
    }


# ---------------------------------------------------------------------------
# build_multi_turn_convos: video path correctness
# ---------------------------------------------------------------------------


class TestBuildMultiTurnConvosVideoPath:
    """Video path should equal the filename field (no double-prefix)."""

    def test_video_path_no_double_prefix(self, tmp_path):
        """The video field should use filename as-is, not prepend source."""
        # Create a fake video so the existence check passes
        video_rel = "val/000030/rgb__0.mp4"
        video_full = tmp_path / video_rel
        video_full.parent.mkdir(parents=True)
        video_full.touch()

        qa = _make_qa_entry(filename=video_rel, source="my_dataset")
        chunks = [[qa]]

        convos, missing = build_multi_turn_convos(
            chunks, "mc", dataset_dir=str(tmp_path)
        )

        assert len(convos) == 1
        assert convos[0]["video"] == video_rel

    def test_no_source_in_video_path(self, tmp_path):
        """The source/dataset_version should NOT appear in the video path."""
        video_rel = "val/000030/rgb__0.mp4"
        video_full = tmp_path / video_rel
        video_full.parent.mkdir(parents=True)
        video_full.touch()

        qa = _make_qa_entry(filename=video_rel, source="feb19_Ultra")
        chunks = [[qa]]

        convos, _ = build_multi_turn_convos(chunks, "mc", dataset_dir=str(tmp_path))

        assert "feb19_Ultra" not in convos[0]["video"]
        assert convos[0]["video"] == video_rel

    def test_missing_video_detected(self, tmp_path):
        """Videos that don't exist should go to missing_convos."""
        qa = _make_qa_entry(filename="val/000030/rgb__0.mp4")
        chunks = [[qa]]

        convos, missing = build_multi_turn_convos(
            chunks, "mc", dataset_dir=str(tmp_path)
        )

        assert len(convos) == 0
        assert len(missing) == 1

    def test_video_version_substitution(self, tmp_path):
        """Changing video_version should replace the camera name in the path."""
        video_rel = "val/000030/depth__0.mp4"
        video_full = tmp_path / video_rel
        video_full.parent.mkdir(parents=True)
        video_full.touch()

        qa = _make_qa_entry(filename="val/000030/rgb__0.mp4")
        chunks = [[qa]]

        convos, _ = build_multi_turn_convos(
            chunks, "mc", dataset_dir=str(tmp_path), video_version="depth"
        )

        assert len(convos) == 1
        assert convos[0]["video"] == "val/000030/depth__0.mp4"

    def test_released_rgb_filename_is_preserved(self, tmp_path):
        video_rel = "val/000030/raw_navigation_camera__0.mp4"
        video_full = tmp_path / video_rel
        video_full.parent.mkdir(parents=True)
        video_full.touch()

        convos, missing = build_multi_turn_convos(
            [[_make_qa_entry(filename=video_rel)]],
            "mc",
            dataset_dir=str(tmp_path),
            video_version="rgb",
        )

        assert not missing
        assert convos[0]["video"] == video_rel


def test_rgb_rename_preserves_multiturn_chunking():
    canonical_qas = [
        _make_qa_entry("val/000030/rgb__0.mp4", idx=index) for index in range(8)
    ]
    released_qas = [
        _make_qa_entry("val/000030/raw_navigation_camera__0.mp4", idx=index)
        for index in range(8)
    ]

    canonical = group_qas_for_multiturn(
        canonical_qas, max_qa_per_convo=2, group_by_task=True, seed=7
    )
    released = group_qas_for_multiturn(
        released_qas, max_qa_per_convo=2, group_by_task=True, seed=7
    )

    canonical_indices = [
        [qa["idx"] for qa in chunk] for chunk in canonical["obj_count"]
    ]
    released_indices = [[qa["idx"] for qa in chunk] for chunk in released["obj_count"]]
    assert canonical_indices == released_indices


def test_released_rgb_filename_supports_modality_substitution(tmp_path):
    video_rel = "val/000030/depth__0.mp4"
    video_full = tmp_path / video_rel
    video_full.parent.mkdir(parents=True)
    video_full.touch()

    convos, missing = build_multi_turn_convos(
        [[_make_qa_entry("val/000030/raw_navigation_camera__0.mp4")]],
        "mc",
        dataset_dir=str(tmp_path),
        video_version="depth",
    )

    assert not missing
    assert convos[0]["video"] == video_rel


# ---------------------------------------------------------------------------
# combine → format integration
# ---------------------------------------------------------------------------


class TestCombineFormatIntegration:
    """Test that combine output fed to format produces correct paths."""

    def test_combine_then_format_paths(self, tmp_path):
        """Simulate the combine stage output and verify format stage paths."""
        from sims.qa.qa_combine_mp import process_file_chunk

        # Create a per-video QA file (as QA gen stage would produce)
        scene_dir = tmp_path / "dataset" / "val" / "000030"
        scene_dir.mkdir(parents=True)

        raw_qa = {
            "idx": 0,
            "type": "v0",
            "filename": "000030/rgb__0.mp4",
            "source": "dataset",
            "task": "obj_count",
            "question": "How many chairs?",
            "gt_answer": "2",
            "mc_question": "How many chairs?\nA. 1\nB. 2",
            "mc_answer": "B",
            "mc_choices": ["A. 1", "B. 2"],
        }

        qa_file = scene_dir / "qa_pairs_obj_count__0.jsonl"
        qa_file.write_text(json.dumps(raw_qa) + "\n")

        # Run combine's process_file_chunk
        combined = process_file_chunk([str(qa_file)], "val")

        assert len(combined) == 1
        assert combined[0]["filename"] == "val/000030/rgb__0.mp4"

        # Create video file so existence check passes
        video = tmp_path / "dataset" / "val" / "000030" / "rgb__0.mp4"
        video.touch()

        # Run format's build_multi_turn_convos
        chunks = [[combined[0]]]
        convos, missing = build_multi_turn_convos(
            chunks, "mc", dataset_dir=str(tmp_path / "dataset")
        )

        assert len(convos) == 1
        # The video path should be relative to dataset_dir, no double val/ prefix
        assert convos[0]["video"] == "val/000030/rgb__0.mp4"
        assert not convos[0]["video"].startswith("dataset/val/")

    def test_writer_output_matches_pipeline_completion_path(self, tmp_path):
        dataset_dir = tmp_path / "dataset"
        scene_dir = dataset_dir / "val" / "000030"
        scene_dir.mkdir(parents=True)
        (scene_dir / "rgb__0.mp4").touch()

        combined_file = dataset_dir / "val" / "combined_qa_pairs.jsonl"
        qa = _make_qa_entry(
            filename="val/000030/rgb__0.mp4",
            source="dataset",
        )
        combined_file.write_text(json.dumps(qa) + "\n")

        create_multiturn_jsonl(
            dataset_dir=str(dataset_dir),
            input_filename="combined_qa_pairs.jsonl",
            output_subdir="qas",
            output_filename_base="",
            split="val",
            video_version="rgb",
            max_qa_per_convo=1,
            group_by_task=True,
        )

        assert _is_format_complete(dataset_dir, "val", "rgb") is True

    def test_missing_video_fails_before_replacing_outputs(self, tmp_path):
        dataset_dir = tmp_path / "dataset"
        split_dir = dataset_dir / "val"
        split_dir.mkdir(parents=True)
        qa = _make_qa_entry(
            filename="val/000030/rgb__0.mp4",
            source="dataset",
        )
        (split_dir / "combined_qa_pairs.jsonl").write_text(json.dumps(qa) + "\n")

        output_dir = dataset_dir / "qas" / "val" / "rgb"
        output_dir.mkdir(parents=True)
        existing = output_dir / "mt1_obj_count_mc.jsonl"
        existing.write_text("previous valid output\n")

        with pytest.raises(FileNotFoundError, match="--allow_partial"):
            create_multiturn_jsonl(
                dataset_dir=str(dataset_dir),
                input_filename="combined_qa_pairs.jsonl",
                output_subdir="qas",
                output_filename_base="",
                split="val",
                video_version="rgb",
                group_by_task=True,
            )

        assert existing.read_text() == "previous valid output\n"
        assert not (output_dir / "missing_mt1_obj_count_mc.jsonl").exists()

    def test_allow_partial_writes_missing_video_manifest(self, tmp_path):
        dataset_dir = tmp_path / "dataset"
        split_dir = dataset_dir / "val"
        split_dir.mkdir(parents=True)
        qa = _make_qa_entry(
            filename="val/000030/rgb__0.mp4",
            source="dataset",
        )
        (split_dir / "combined_qa_pairs.jsonl").write_text(json.dumps(qa) + "\n")

        create_multiturn_jsonl(
            dataset_dir=str(dataset_dir),
            input_filename="combined_qa_pairs.jsonl",
            output_subdir="qas",
            output_filename_base="",
            split="val",
            video_version="rgb",
            group_by_task=True,
            allow_partial=True,
        )

        output_dir = dataset_dir / "qas" / "val" / "rgb"
        assert (output_dir / "mt1_obj_count_mc.jsonl").read_text() == ""
        missing = output_dir / "missing_mt1_obj_count_mc.jsonl"
        assert json.loads(missing.read_text())["video"] == qa["filename"]

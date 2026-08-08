"""Tests for sims.pipeline — unified QA dataset generation pipeline."""

from argparse import Namespace
from unittest.mock import MagicMock, patch

import pytest

from sims.pipeline import (
    STAGES,
    _is_combine_complete,
    _is_format_complete,
    _is_metadata_complete,
    _is_qa_complete,
    main,
    parse_args,
    run_combine,
    run_format,
    run_metadata,
    run_qa,
)


# ---------------------------------------------------------------------------
# parse_args
# ---------------------------------------------------------------------------


class TestParseArgs:
    def test_required_only(self):
        args = parse_args(["--dataset-dir", "/tmp/data"])
        assert args.dataset_dir == "/tmp/data"
        assert args.split == "val"
        assert args.stages is None
        assert args.question_types == ["obj_count"]
        assert args.num_questions_per_video == 5
        assert args.num_workers == 16
        assert args.seed == 42
        assert args.allow_partial is False
        assert args.num_workers_per_gpu == 4
        assert args.video_version == "rgb"
        assert args.answer_mode == "mc"
        assert args.max_qa_per_convo == 1
        assert args.group_by_task is True
        assert args.skip_completed is False
        assert args.overwrite_metadata is False

    def test_custom_split(self):
        args = parse_args(["--dataset-dir", "d", "--split", "train"])
        assert args.split == "train"

    def test_custom_stages(self):
        args = parse_args(["--dataset-dir", "d", "--stages", "metadata", "qa"])
        assert args.stages == ["metadata", "qa"]

    def test_single_stage(self):
        args = parse_args(["--dataset-dir", "d", "--stages", "combine"])
        assert args.stages == ["combine"]

    def test_multiple_question_types(self):
        args = parse_args(
            ["--dataset-dir", "d", "--question-types", "obj_count", "temporal_order_5"]
        )
        assert args.question_types == ["obj_count", "temporal_order_5"]

    def test_skip_completed(self):
        args = parse_args(["--dataset-dir", "d", "--skip-completed"])
        assert args.skip_completed is True

    def test_overwrite_metadata(self):
        args = parse_args(["--dataset-dir", "d", "--overwrite-metadata"])
        assert args.overwrite_metadata is True

    def test_no_group_by_task(self):
        args = parse_args(["--dataset-dir", "d", "--no-group-by-task"])
        assert args.group_by_task is False

    def test_custom_format_args(self):
        args = parse_args(
            [
                "--dataset-dir",
                "d",
                "--video-version",
                "depth",
                "--max-qa-per-convo",
                "3",
                "--answer-mode",
                "oe",
            ]
        )
        assert args.video_version == "depth"
        assert args.max_qa_per_convo == 3
        assert args.answer_mode == "oe"

    def test_missing_required_exits(self):
        with pytest.raises(SystemExit):
            parse_args([])

    def test_invalid_stage_exits(self):
        with pytest.raises(SystemExit):
            parse_args(["--dataset-dir", "d", "--stages", "nonexistent"])

    def test_invalid_question_type_exits(self):
        with pytest.raises(SystemExit):
            parse_args(["--dataset-dir", "d", "--question-types", "misspelled"])

    def test_invalid_video_version_exits(self):
        with pytest.raises(SystemExit):
            parse_args(["--dataset-dir", "d", "--video-version", "wireframe"])

    @pytest.mark.parametrize(
        "option,value",
        [
            ("--num-workers-per-gpu", "0"),
            ("--num-questions-per-video", "0"),
            ("--num-workers", "-1"),
            ("--max-qa-per-convo", "0"),
        ],
    )
    def test_nonpositive_counts_exit(self, option, value):
        with pytest.raises(SystemExit):
            parse_args(["--dataset-dir", "d", option, value])


# ---------------------------------------------------------------------------
# Completion checks
# ---------------------------------------------------------------------------


class TestCompletionChecks:
    def test_metadata_incomplete(self, tmp_path):
        assert _is_metadata_complete(tmp_path) is False

    def test_metadata_complete_flat(self, tmp_path):
        (tmp_path / "house_spatial_metadata.pkl").touch()
        assert _is_metadata_complete(tmp_path) is False

    def test_metadata_complete_nested(self, tmp_path):
        subdir = tmp_path / "000001"
        subdir.mkdir()
        (subdir / "spatial_metadata.json").touch()
        assert _is_metadata_complete(tmp_path) is True

    def test_metadata_requires_every_scene(self, tmp_path):
        complete = tmp_path / "000001"
        complete.mkdir()
        (complete / "spatial_metadata.json").touch()
        (tmp_path / "000002").mkdir()
        assert _is_metadata_complete(tmp_path) is False

    def test_qa_incomplete(self, tmp_path):
        assert _is_qa_complete(tmp_path) is False

    def test_qa_complete(self, tmp_path):
        scene_dir = tmp_path / "000001"
        scene_dir.mkdir()
        (scene_dir / "spatial_metadata.json").touch()
        (scene_dir / "offline_annos__0.jsonl").touch()
        (scene_dir / "rgb__0.mp4").touch()
        (scene_dir / "qa_pairs_obj_count__0.jsonl").touch()
        assert _is_qa_complete(tmp_path, ["obj_count"]) is True

    def test_qa_complete_accepts_released_rgb_filename(self, tmp_path):
        scene_dir = tmp_path / "000001"
        scene_dir.mkdir()
        (scene_dir / "spatial_metadata.json").touch()
        (scene_dir / "offline_annos__0.jsonl").touch()
        (scene_dir / "raw_navigation_camera__0.mp4").touch()
        (scene_dir / "qa_pairs_obj_count__0.jsonl").touch()

        assert _is_qa_complete(tmp_path, ["obj_count"]) is True

    def test_qa_incomplete_when_both_rgb_names_exist(self, tmp_path):
        scene_dir = tmp_path / "000001"
        scene_dir.mkdir()
        (scene_dir / "spatial_metadata.json").touch()
        (scene_dir / "offline_annos__0.jsonl").touch()
        (scene_dir / "rgb__0.mp4").touch()
        (scene_dir / "raw_navigation_camera__0.mp4").touch()
        (scene_dir / "qa_pairs_obj_count__0.jsonl").touch()

        assert _is_qa_complete(tmp_path, ["obj_count"]) is False

    def test_qa_requires_every_requested_question_type(self, tmp_path):
        scene_dir = tmp_path / "000001"
        scene_dir.mkdir()
        (scene_dir / "spatial_metadata.json").touch()
        (scene_dir / "offline_annos__0.jsonl").touch()
        (scene_dir / "rgb__0.mp4").touch()
        (scene_dir / "qa_pairs_obj_count__0.jsonl").touch()
        assert _is_qa_complete(tmp_path, ["obj_count", "temporal_order_2"]) is False

    @pytest.mark.parametrize("missing", ["metadata", "annotations", "video"])
    def test_qa_incomplete_when_any_scene_prerequisite_is_missing(
        self, tmp_path, missing
    ):
        complete = tmp_path / "000001"
        complete.mkdir()
        (complete / "spatial_metadata.json").touch()
        (complete / "offline_annos__0.jsonl").touch()
        (complete / "rgb__0.mp4").touch()
        (complete / "qa_pairs_obj_count__0.jsonl").touch()

        incomplete = tmp_path / "000002"
        incomplete.mkdir()
        if missing != "metadata":
            (incomplete / "spatial_metadata.json").touch()
        if missing != "annotations":
            (incomplete / "offline_annos__0.jsonl").touch()
        if missing != "video":
            (incomplete / "rgb__0.mp4").touch()
        # A stale QA file must not make the invalid scene skippable.
        (incomplete / "qa_pairs_obj_count__0.jsonl").touch()

        assert _is_qa_complete(tmp_path, ["obj_count"]) is False

    def test_combine_incomplete(self, tmp_path):
        assert _is_combine_complete(tmp_path) is False

    def test_combine_complete(self, tmp_path):
        (tmp_path / "combined_qa_pairs.jsonl").write_text("{}\n")
        assert _is_combine_complete(tmp_path) is True

    def test_format_incomplete_no_dir(self, tmp_path):
        assert _is_format_complete(tmp_path, "val", "rgb") is False

    def test_format_incomplete_empty_dir(self, tmp_path):
        (tmp_path / "qas" / "val" / "rgb").mkdir(parents=True)
        assert _is_format_complete(tmp_path, "val", "rgb") is False

    def test_format_complete(self, tmp_path):
        fmt_dir = tmp_path / "qas" / "val" / "rgb"
        fmt_dir.mkdir(parents=True)
        (fmt_dir / "output.jsonl").write_text("{}\n")
        assert _is_format_complete(tmp_path, "val", "rgb") is True

    def test_format_respects_video_version(self, tmp_path):
        fmt_dir = tmp_path / "qas" / "val" / "depth"
        fmt_dir.mkdir(parents=True)
        (fmt_dir / "output.jsonl").write_text("{}\n")
        assert _is_format_complete(tmp_path, "val", "depth") is True
        assert _is_format_complete(tmp_path, "val", "rgb") is False

    def test_format_ignores_missing_video_report(self, tmp_path):
        fmt_dir = tmp_path / "qas" / "val" / "rgb"
        fmt_dir.mkdir(parents=True)
        (fmt_dir / "missing_output.jsonl").write_text("{}\n")
        assert _is_format_complete(tmp_path, "val", "rgb") is False


# ---------------------------------------------------------------------------
# Stage runners (mocked)
# ---------------------------------------------------------------------------


def _make_args(**overrides):
    defaults = dict(
        dataset_dir="/tmp/test_dataset",
        split="val",
        overwrite_metadata=False,
        num_workers_per_gpu=4,
        question_types=["obj_count"],
        num_questions_per_video=5,
        num_workers=16,
        video_version="rgb",
        answer_mode="mc",
        max_qa_per_convo=1,
        group_by_task=True,
        seed=42,
        allow_partial=False,
        skip_completed=False,
        stages=None,
    )
    defaults.update(overrides)
    return Namespace(**defaults)


class TestStageRunners:
    @patch("sims.qa.spatial_metadata_gen_mp.main")
    def test_run_metadata(self, mock_main):
        args = _make_args(overwrite_metadata=True, num_workers_per_gpu=2)
        run_metadata(args)
        mock_main.assert_called_once_with(
            "/tmp/test_dataset",
            split="val",
            overwrite=True,
            num_workers_per_gpu=2,
            allow_partial=False,
        )

    @patch("sims.qa.spatial_qa_gen.generate_qa_for_dataset")
    def test_run_qa_single_type(self, mock_gen):
        args = _make_args(question_types=["obj_count"])
        run_qa(args)
        mock_gen.assert_called_once_with(
            "/tmp/test_dataset",
            source="test_dataset",
            split="val",
            num_questions_per_video=5,
            question_type="obj_count",
            num_workers=16,
            seed=42,
            allow_partial=False,
        )

    @patch("sims.qa.spatial_qa_gen.generate_qa_for_dataset")
    def test_run_qa_multiple_types(self, mock_gen):
        args = _make_args(question_types=["obj_count", "temporal_order_5", "n_rooms"])
        run_qa(args)
        assert mock_gen.call_count == 3
        types_called = [c.kwargs["question_type"] for c in mock_gen.call_args_list]
        assert types_called == ["obj_count", "temporal_order_5", "n_rooms"]

    @patch("sims.qa.qa_combine_mp.combine_qa_files")
    def test_run_combine(self, mock_combine):
        args = _make_args()
        run_combine(args)
        mock_combine.assert_called_once_with(
            "/tmp/test_dataset",
            "val",
            output_filename="combined_qa_pairs.jsonl",
            question_types=["obj_count"],
        )

    @patch("sims.qa.qa_to_ov_train_multiturn.create_multiturn_jsonl")
    def test_run_format(self, mock_fmt):
        args = _make_args(
            video_version="depth", max_qa_per_convo=3, group_by_task=False
        )
        run_format(args)
        mock_fmt.assert_called_once_with(
            dataset_dir="/tmp/test_dataset",
            input_filename="combined_qa_pairs.jsonl",
            output_subdir="qas",
            output_filename_base="",
            split="val",
            question_mode="mc",
            video_version="depth",
            max_qa_per_convo=3,
            group_by_task=False,
            seed=42,
            question_types=["obj_count"],
            allow_partial=False,
        )


# ---------------------------------------------------------------------------
# main() orchestration
# ---------------------------------------------------------------------------


class TestMain:
    @patch("sims.pipeline._STAGE_RUNNERS")
    def test_runs_all_stages_by_default(self, mock_runners, tmp_path):
        split_dir = tmp_path / "val"
        split_dir.mkdir()

        mock_fns = {s: MagicMock() for s in STAGES}
        mock_runners.__getitem__ = lambda self, key: mock_fns[key]

        main(["--dataset-dir", str(tmp_path), "--split", "val"])

        for stage in STAGES:
            mock_fns[stage].assert_called_once()

    @patch("sims.pipeline._STAGE_RUNNERS")
    def test_runs_only_selected_stages(self, mock_runners, tmp_path):
        split_dir = tmp_path / "val"
        split_dir.mkdir()

        mock_fns = {s: MagicMock() for s in STAGES}
        mock_runners.__getitem__ = lambda self, key: mock_fns[key]

        main(
            [
                "--dataset-dir",
                str(tmp_path),
                "--split",
                "val",
                "--stages",
                "metadata",
                "combine",
            ]
        )

        mock_fns["metadata"].assert_called_once()
        mock_fns["combine"].assert_called_once()
        mock_fns["qa"].assert_not_called()
        mock_fns["format"].assert_not_called()

    @patch("sims.pipeline._STAGE_RUNNERS")
    def test_rebuilds_cheap_combine_stage_with_skip_completed(
        self, mock_runners, tmp_path
    ):
        split_dir = tmp_path / "val"
        split_dir.mkdir()

        # Mark combine as complete
        (split_dir / "combined_qa_pairs.jsonl").write_text("{}\n")

        mock_fns = {s: MagicMock() for s in STAGES}
        mock_runners.__getitem__ = lambda self, key: mock_fns[key]

        main(
            [
                "--dataset-dir",
                str(tmp_path),
                "--split",
                "val",
                "--stages",
                "combine",
                "--skip-completed",
            ]
        )

        mock_fns["combine"].assert_called_once()

    @patch("sims.pipeline._STAGE_RUNNERS")
    def test_does_not_skip_without_flag(self, mock_runners, tmp_path):
        split_dir = tmp_path / "val"
        split_dir.mkdir()

        # Mark combine as complete, but don't pass --skip-completed
        (split_dir / "combined_qa_pairs.jsonl").write_text("{}\n")

        mock_fns = {s: MagicMock() for s in STAGES}
        mock_runners.__getitem__ = lambda self, key: mock_fns[key]

        main(["--dataset-dir", str(tmp_path), "--split", "val", "--stages", "combine"])

        mock_fns["combine"].assert_called_once()

    def test_exits_on_missing_split_dir(self, tmp_path):
        with pytest.raises(SystemExit):
            main(["--dataset-dir", str(tmp_path), "--split", "nonexistent"])

    def test_objaverse_path_error_uses_cli_error(self, tmp_path, capsys):
        with pytest.raises(SystemExit) as error:
            main(
                [
                    "--dataset-dir",
                    str(tmp_path),
                    "--objaverse-dir",
                    str(tmp_path / "missing-assets"),
                ]
            )

        assert error.value.code == 2
        assert "Objaverse asset directory does not exist" in capsys.readouterr().err

    def test_objaverse_environment_error_uses_cli_error(
        self, tmp_path, monkeypatch, capsys
    ):
        monkeypatch.setenv("OBJAVERSE_DATA_DIR", str(tmp_path / "missing-assets"))

        with pytest.raises(SystemExit) as error:
            main(["--dataset-dir", str(tmp_path), "--stages", "metadata"])

        assert error.value.code == 2
        assert "Objaverse asset directory does not exist" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# chunks() from qa_combine_mp
# ---------------------------------------------------------------------------


class TestChunks:
    def test_even_split(self):
        from sims.qa.qa_combine_mp import chunks

        assert list(chunks([1, 2, 3, 4], 2)) == [[1, 2], [3, 4]]

    def test_uneven_split(self):
        from sims.qa.qa_combine_mp import chunks

        assert list(chunks([1, 2, 3, 4, 5], 2)) == [[1, 2], [3, 4], [5]]

    def test_empty_list(self):
        from sims.qa.qa_combine_mp import chunks

        assert list(chunks([], 3)) == []

    def test_chunk_larger_than_list(self):
        from sims.qa.qa_combine_mp import chunks

        assert list(chunks([1, 2], 10)) == [[1, 2]]

    def test_chunk_size_one(self):
        from sims.qa.qa_combine_mp import chunks

        assert list(chunks([1, 2, 3], 1)) == [[1], [2], [3]]

    def test_single_element(self):
        from sims.qa.qa_combine_mp import chunks

        assert list(chunks([42], 5)) == [[42]]


# ---------------------------------------------------------------------------
# STAGES constant
# ---------------------------------------------------------------------------


def test_stages_order():
    assert STAGES == ["metadata", "qa", "combine", "format"]


def test_combine_rejects_malformed_json(tmp_path):
    from sims.qa.qa_combine_mp import process_file_chunk

    malformed = tmp_path / "qa_pairs_obj_count__0.jsonl"
    malformed.write_text("{not-json}\n")

    with pytest.raises(ValueError, match="Invalid JSON"):
        process_file_chunk([str(malformed)], "val")

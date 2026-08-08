"""Unified pipeline for QA dataset generation.

Chains the 4 QA generation stages:
  1. metadata  - Extract spatial metadata from houses
  2. qa        - Generate question-answer pairs
  3. combine   - Merge per-video QA files into single JSONL
  4. format    - Convert to training format

Usage:
    sims-v qa --dataset-dir <path> --split val
    sims-v qa --dataset-dir <path> --stages metadata qa
    sims-v qa --dataset-dir <path> --skip-completed
"""

import argparse
import logging
import sys
from pathlib import Path

from sims.data_generation.arg_parsers import positive_int
from sims.video_paths import index_rgb_video_filenames

logger = logging.getLogger(__name__)

STAGES = ["metadata", "qa", "combine", "format"]


# ---------------------------------------------------------------------------
# Completion checks
# ---------------------------------------------------------------------------


def _is_metadata_complete(split_dir):
    scene_dirs = [path for path in split_dir.iterdir() if path.is_dir()]
    return bool(scene_dirs) and all(
        (scene_dir / "spatial_metadata.json").is_file() for scene_dir in scene_dirs
    )


def _is_qa_complete(split_dir, question_types=("obj_count",)):
    scene_dirs = sorted(path for path in split_dir.iterdir() if path.is_dir())
    if not scene_dirs or not question_types:
        return False

    expected_paths = []
    for scene_dir in scene_dirs:
        if not (scene_dir / "spatial_metadata.json").is_file():
            return False

        annotation_indices = {
            path.stem.removeprefix("offline_annos__")
            for path in scene_dir.glob("offline_annos__*.jsonl")
        }
        try:
            video_indices = set(
                index_rgb_video_filenames(path.name for path in scene_dir.glob("*.mp4"))
            )
        except ValueError:
            return False
        if not annotation_indices or annotation_indices != video_indices:
            return False

        for video_index in sorted(annotation_indices):
            expected_paths.extend(
                scene_dir / f"qa_pairs_{question_type}__{video_index}.jsonl"
                for question_type in question_types
            )

    return all(path.is_file() for path in expected_paths)


def _is_combine_complete(split_dir):
    output_path = split_dir / "combined_qa_pairs.jsonl"
    return output_path.is_file() and output_path.stat().st_size > 0


def _is_format_complete(dataset_dir, split, video_version):
    fmt_dir = dataset_dir / "qas" / split / video_version
    return fmt_dir.is_dir() and any(
        path.is_file()
        and path.suffix == ".jsonl"
        and not path.name.startswith("missing_")
        and path.stat().st_size > 0
        for path in fmt_dir.iterdir()
    )


_COMPLETION_CHECKS = {
    "metadata": lambda dataset_dir, split_dir, **kw: _is_metadata_complete(split_dir),
    "qa": lambda dataset_dir, split_dir, **kw: _is_qa_complete(
        split_dir, kw["question_types"]
    ),
    "combine": lambda dataset_dir, split_dir, **kw: _is_combine_complete(split_dir),
    "format": lambda dataset_dir, split_dir, **kw: _is_format_complete(
        dataset_dir, kw["split"], kw["video_version"]
    ),
}


# ---------------------------------------------------------------------------
# Stage runners
# ---------------------------------------------------------------------------


def run_metadata(args):
    from sims.qa.spatial_metadata_gen_mp import main as metadata_main

    metadata_main(
        args.dataset_dir,
        split=args.split,
        overwrite=args.overwrite_metadata,
        num_workers_per_gpu=args.num_workers_per_gpu,
        allow_partial=args.allow_partial,
    )


def run_qa(args):
    from sims.qa.spatial_qa_gen import generate_qa_for_dataset

    for qt in args.question_types:
        logger.info("Generating '%s' questions...", qt)
        generate_qa_for_dataset(
            args.dataset_dir,
            source=Path(args.dataset_dir).name,
            split=args.split,
            num_questions_per_video=args.num_questions_per_video,
            question_type=qt,
            num_workers=args.num_workers,
            seed=args.seed,
            allow_partial=args.allow_partial,
        )


def run_combine(args):
    from sims.qa.qa_combine_mp import combine_qa_files

    combine_qa_files(
        args.dataset_dir,
        args.split,
        output_filename="combined_qa_pairs.jsonl",
        question_types=args.question_types,
    )


def run_format(args):
    from sims.qa.qa_to_ov_train_multiturn import create_multiturn_jsonl

    create_multiturn_jsonl(
        dataset_dir=args.dataset_dir,
        input_filename="combined_qa_pairs.jsonl",
        output_subdir="qas",
        output_filename_base="",
        split=args.split,
        question_mode=args.answer_mode,
        video_version=args.video_version,
        max_qa_per_convo=args.max_qa_per_convo,
        group_by_task=args.group_by_task,
        seed=args.seed,
        question_types=args.question_types,
        allow_partial=args.allow_partial,
    )


_STAGE_RUNNERS = {
    "metadata": run_metadata,
    "qa": run_qa,
    "combine": run_combine,
    "format": run_format,
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser(*, prog=None):
    # Build these choices from the registries that execute them so the public
    # CLI cannot silently accept a misspelled or retired mode.
    from sims.qa.generators import QA_GEN_FNS
    from sims.qa.qa_to_ov_train_multiturn import VERS_TO_VIDFN

    parser = argparse.ArgumentParser(
        prog=prog,
        description="Unified QA dataset generation pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  # Run all stages
  sims-v qa --dataset-dir experiment_output/my_dataset

  # Generate only metadata and QA pairs
  sims-v qa --dataset-dir experiment_output/my_dataset --stages metadata qa

  # Multiple question types
  sims-v qa --dataset-dir experiment_output/my_dataset \\
      --question-types obj_count obj_rel_distance temporal_order_5

  # Skip stages whose output already exists
  sims-v qa --dataset-dir experiment_output/my_dataset --skip-completed
""",
    )

    # Shared
    parser.add_argument(
        "--dataset-dir",
        required=True,
        help="Path to dataset directory",
    )
    parser.add_argument(
        "--objaverse-dir",
        help="Objaverse SIMS asset directory when metadata extraction needs it",
    )
    parser.add_argument(
        "--split", type=str, default="val", help="Dataset split (default: val)"
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=STAGES,
        default=None,
        help="Stages to run (default: all). Choices: %(choices)s",
    )
    parser.add_argument(
        "--skip-completed",
        action="store_true",
        help=(
            "Skip complete metadata and QA stages; combine and format are always "
            "rebuilt. Do not use after changing the QA seed or question count."
        ),
    )

    # Metadata stage
    meta = parser.add_argument_group("metadata stage")
    meta.add_argument(
        "--num-workers-per-gpu",
        type=positive_int,
        default=4,
        help="Workers per GPU for metadata extraction (default: 4)",
    )
    meta.add_argument(
        "--overwrite-metadata",
        action="store_true",
        help="Overwrite existing metadata files",
    )

    # QA stage
    qa = parser.add_argument_group("qa stage")
    qa.add_argument(
        "--question-types",
        nargs="+",
        choices=tuple(QA_GEN_FNS),
        metavar="TYPE",
        default=["obj_count"],
        help="Question types to generate (default: obj_count). Choices: %(choices)s",
    )
    qa.add_argument(
        "--num-questions-per-video",
        type=positive_int,
        default=5,
        help="Questions per video (default: 5)",
    )
    qa.add_argument(
        "--num-workers",
        type=positive_int,
        default=16,
        help="Parallel workers for QA generation (default: 16)",
    )
    qa.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base seed for deterministic QA generation (default: 42)",
    )
    qa.add_argument(
        "--allow-partial",
        action="store_true",
        help="Keep successful outputs if a scene or selected video is missing (default: fail)",
    )

    # Format stage
    fmt = parser.add_argument_group("format stage")
    fmt.add_argument(
        "--answer-mode",
        choices=("mc", "mc_direct", "oe"),
        default="mc",
        help="Formatted answer style (default: mc)",
    )
    fmt.add_argument(
        "--video-version",
        choices=tuple(VERS_TO_VIDFN),
        metavar="VERSION",
        default="rgb",
        help="Video modality (default: rgb). Choices: %(choices)s",
    )
    fmt.add_argument(
        "--max-qa-per-convo",
        type=positive_int,
        default=1,
        help="QA pairs per conversation (default: 1)",
    )
    fmt.add_argument(
        "--group-by-task",
        action="store_true",
        default=True,
        help="Separate output by task type (default: True)",
    )
    fmt.add_argument(
        "--no-group-by-task",
        action="store_false",
        dest="group_by_task",
        help="Don't separate output by task type",
    )

    return parser


def parse_args(argv=None, *, prog=None):
    return build_parser(prog=prog).parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv=None, *, prog=None):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = build_parser(prog=prog)
    args = parser.parse_args(argv)
    stages = args.stages or STAGES
    from sims.data_generation.paths import resolve_objaverse_data_dir

    try:
        if "metadata" in stages:
            objaverse_dir = resolve_objaverse_data_dir(args.objaverse_dir)
            if objaverse_dir is not None:
                resolve_objaverse_data_dir(objaverse_dir, required=True)
        elif args.objaverse_dir:
            resolve_objaverse_data_dir(args.objaverse_dir)
    except FileNotFoundError as error:
        parser.error(str(error))
    dataset_dir = Path(args.dataset_dir)
    split_dir = dataset_dir / args.split

    if not split_dir.exists():
        logger.error("Split directory does not exist: %s", split_dir)
        sys.exit(1)

    logger.info("Pipeline: dataset_dir=%s  split=%s", args.dataset_dir, args.split)
    logger.info("Stages: %s", " → ".join(stages))

    for stage in stages:
        # Combine and format are cheap derived stages.  Always rebuild them so
        # a rerun with a narrower question-type selection cannot reuse stale
        # records or formatter outputs.
        if args.skip_completed and stage not in {"combine", "format"}:
            check = _COMPLETION_CHECKS[stage]
            if check(
                dataset_dir,
                split_dir,
                split=args.split,
                video_version=args.video_version,
                question_types=args.question_types,
            ):
                logger.info("[%s] skipped (output exists)", stage)
                continue

        logger.info("[%s] starting...", stage)
        _STAGE_RUNNERS[stage](args)
        logger.info("[%s] done", stage)

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()

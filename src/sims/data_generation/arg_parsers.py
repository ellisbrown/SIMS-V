import argparse
import math


QUALITY_LEVELS = (
    "Very Low",
    "Low",
    "Medium",
    "MediumCloseFitShadows",
    "High",
    "Very High",
    "Ultra",
    "High WebGL",
)

EXTRA_VIDEO_MODALITIES = (
    "depth",
    "semantic_seg",
    "instance_seg",
    "edge",
    "colored_edge",
    "non_overlapping_colored_edge",
    "mean_mask_overlay",
    "masked_background",
)


def resolve_extra_video_modalities(selected):
    """Return selected modalities once, in stable output order."""
    selected = set(selected)
    if "all" in selected:
        return EXTRA_VIDEO_MODALITIES
    return tuple(name for name in EXTRA_VIDEO_MODALITIES if name in selected)


def probability(value):
    """Parse a floating-point probability."""
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a number between 0 and 1") from error
    if not 0 <= parsed <= 1:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return parsed


def positive_int(value):
    """Parse a positive integer."""
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive integer") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def nonnegative_float(value):
    """Parse a nonnegative floating-point value."""
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be finite and nonnegative") from error
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("must be finite and nonnegative")
    return parsed


def positive_float(value):
    """Parse a positive floating-point value."""
    parsed = nonnegative_float(value)
    if parsed == 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def add_generate_arguments(parser):
    parser.add_argument(
        "--dataset-dir",
        required=True,
        help="Directory in which to write the generated dataset.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "val", "test"),
        default="val",
        help="Dataset split to generate (default: val).",
    )
    parser.add_argument(
        "--house-dataset",
        choices=("procthor", "objaverse"),
        default="procthor",
        help="Source of procedural houses (default: procthor).",
    )
    parser.add_argument(
        "--objaverse-dir",
        help=(
            "Objaverse SIMS asset directory. Defaults to OBJAVERSE_DATA_DIR, "
            "then ./objaverse_sims when present."
        ),
    )
    parser.add_argument(
        "--trajectories-per-house",
        type=positive_int,
        default=1,
        help="Walkthroughs to record per house (default: 1).",
    )
    parser.add_argument(
        "--max-houses",
        type=positive_int,
        default=1,
        help="Maximum candidate houses to process (default: 1).",
    )
    parser.add_argument(
        "--max-steps",
        type=positive_int,
        default=1000,
        help="Maximum agent actions per walkthrough (default: 1000).",
    )
    parser.add_argument(
        "--workers",
        type=positive_int,
        default=None,
        help="Generation workers (default: inferred from GPUs and memory).",
    )
    parser.add_argument(
        "--resolution-scale",
        type=positive_float,
        default=1.0,
        help=(
            "Scale the default 396x224 controller render; raw cameras crop "
            "6 pixels per horizontal edge (default: 1)."
        ),
    )
    parser.add_argument(
        "--quality",
        choices=QUALITY_LEVELS,
        default="Ultra",
        help="AI2-THOR render quality (default: Ultra).",
    )
    parser.add_argument(
        "--extra-video-modalities",
        nargs="+",
        choices=(*EXTRA_VIDEO_MODALITIES, "all"),
        default=(),
        metavar="MODALITY",
        help=(
            "Additional videos to render alongside the primary RGB walkthrough "
            "(default: none). Use 'all' for every available modality. "
            "Choices: %(choices)s."
        ),
    )
    parser.add_argument(
        "--material-randomization-probability",
        type=probability,
        default=0.0,
        help="Probability of randomizing a house's materials (default: 0).",
    )
    parser.add_argument(
        "--rotation-noise-std-degrees",
        type=nonnegative_float,
        default=0.0,
        help="Gaussian rotation-noise standard deviation (default: 0).",
    )
    return parser


def get_arg_parser_for_offline_datagen(prog=None):
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Generate simulated house walkthroughs and sensor recordings.",
    )
    return add_generate_arguments(parser)


def parse_args_for_offline_datagen(argv=None, prog=None):
    return get_arg_parser_for_offline_datagen(prog=prog).parse_args(argv)

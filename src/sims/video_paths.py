"""Canonical and legacy SIMS video-path handling."""

RGB_VIDEO_STEM = "rgb"
LEGACY_RGB_VIDEO_STEMS = ("raw_navigation_camera",)
RGB_VIDEO_STEMS = (RGB_VIDEO_STEM, *LEGACY_RGB_VIDEO_STEMS)

VIDEO_VERSION_TO_STEM = {
    "rgb": RGB_VIDEO_STEM,
    "depth": "depth",
    "edge": "edge",
    "colored_edge": "colored_edge",
    "colored_edge_no": "non_overlapping_colored_edge",
    "semantic_seg": "semantic_seg",
    "instance_seg": "instance_seg",
    "mean_mask": "mean_mask_overlay",
    "masked_bg": "masked_background",
}


def _split_rgb_video_path(filename):
    normalized = str(filename).replace("\\", "/")
    basename = normalized.rsplit("/", 1)[-1]
    for stem in RGB_VIDEO_STEMS:
        prefix = f"{stem}__"
        if basename.startswith(prefix) and basename.endswith(".mp4"):
            index = basename[len(prefix) : -len(".mp4")]
            if index:
                return normalized, basename, stem, index
    return None


def index_rgb_video_filenames(filenames):
    """Map scene-local trajectory indices to RGB basenames, including legacy names."""
    by_index = {}
    for filename in sorted(filenames):
        parsed = _split_rgb_video_path(filename)
        if parsed is None:
            continue
        _, basename, _, index = parsed
        previous = by_index.get(index)
        if previous is not None and previous != basename:
            raise ValueError(
                f"multiple RGB videos for trajectory {index!r}: "
                f"{previous!r} and {basename!r}"
            )
        by_index[index] = basename
    return by_index


def video_path_for_version(filename, video_version):
    """Select a modality while preserving canonical or legacy RGB input paths."""
    if video_version not in VIDEO_VERSION_TO_STEM:
        raise ValueError(f"Invalid video_version: {video_version}")

    original = str(filename)
    parsed = _split_rgb_video_path(original)
    if parsed is None:
        raise ValueError(f"Not a recognized RGB video path: {filename}")
    normalized, basename, _, index = parsed
    if video_version == "rgb":
        return original

    target_basename = f"{VIDEO_VERSION_TO_STEM[video_version]}__{index}.mp4"
    return f"{normalized[: -len(basename)]}{target_basename}"


def qa_seed_video_identity(filename):
    """Keep QA sampling stable across the public RGB filename rename."""
    normalized = str(filename).replace("\\", "/")
    parsed = _split_rgb_video_path(normalized)
    if parsed is None:
        return normalized
    _, basename, stem, index = parsed
    if stem in LEGACY_RGB_VIDEO_STEMS:
        return normalized

    legacy_basename = f"{LEGACY_RGB_VIDEO_STEMS[0]}__{index}.mp4"
    return f"{normalized[: -len(basename)]}{legacy_basename}"

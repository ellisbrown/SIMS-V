import argparse
import json
import os
import random
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm

from sims.qa.generators.common import stable_seed
from sims.qa.spatial_metadata_gen import METADATA_FNAME
from sims.video_paths import index_rgb_video_filenames, qa_seed_video_identity

# Re-export everything from generators for backward compatibility
from sims.qa.generators import (  # noqa: F401
    QA_GEN_FNS,
    ALL_OBJECTS,
    ALL_OBJECTS_CLEAN,
    MIN_NUM_FRAMES,
    VSI_OBJECTS,
    VSI_OBJECTS_CLEAN,
    calc_3d_bbox_distance_between_objects,
    calculate_relative_direction,
    clean_obj_type,
    gen_binary_qa,
    gen_binary_qas,
    gen_count_options,
    gen_distance_options,
    gen_house_size_est_qas,
    gen_mc_question,
    gen_n_rooms_qas,
    gen_obj_abs_dist_qa,
    gen_obj_abs_dist_qas,
    gen_obj_count_qas,
    gen_obj_rel_dir_qa,
    gen_obj_rel_dir_qas,
    gen_obj_rel_dist_qa,
    gen_obj_rel_dist_qas,
    gen_obj_size_est_qas,
    gen_temporal_order_qa,
    gen_temporal_order_qas,
    gen_temporal_rel_qa,
    gen_temporal_rel_qas,
    generate_qa_batch,
)


# >>>>> GENERATION INFRA <<<<<< #

DEFAULT_VISIBILITY_THRESHOLD = 0.05


class IneligibleVideoError(ValueError):
    """A valid video cannot support QA generation (for example, it is too short)."""


class QAProcessingError(RuntimeError):
    """A video or scene could not be processed because its inputs are invalid."""


def preprocess_salient_objects(
    jsonl_file_path,
    metadata_json_path,
    visibility_threshold=DEFAULT_VISIBILITY_THRESHOLD,
    ignore_types=["Wall", "Floor"],
):
    """Collect trajectory-visible objects used by the paper QA generators.

    ``pct_pixels`` is a fraction of the frame.  An instance is salient if its
    mask covers strictly more than ``visibility_threshold`` in any frame; the
    release default therefore corresponds to more than five percent.
    """
    data = []
    with open(jsonl_file_path, "r") as file:
        for line in file:
            data.append(json.loads(line))

    if len(data) < MIN_NUM_FRAMES:
        raise IneligibleVideoError(f"Insufficient frames in video: {len(data)}")

    with open(metadata_json_path, "r") as file:
        spatial_metadata = json.load(file)

    # Counters and mappings to track objects
    salient_objects_n_frames = (
        Counter()
    )  # Counts how many times each object appears in frames
    salient_obj_idx_map = defaultdict(
        list
    )  # Maps object keys to the frame indices they appear in
    synset_map = defaultdict(set)  # Maps synsets to object keys
    object_info = {}  # Stores object information (e.g., type, synset, color)
    max_visibility_map = defaultdict(float)

    # Iterate over frames and identify salient objects
    for idx, frame in enumerate(data):
        for key, obj in frame["objects"].items():
            if obj["object_type"] in ignore_types:
                continue
            if obj["pct_pixels"] > visibility_threshold:
                salient_objects_n_frames[key] += 1
                salient_obj_idx_map[key].append(idx)
                synset_map[obj["synset"]].add(key)
                if key not in object_info:
                    clean_object_type = obj["object_type"].replace(
                        "Obja", ""
                    )  # Objaverse synthetic types use an "Obja" prefix.
                    object_info[key] = {
                        "object_type": clean_object_type,
                        "synset": obj["synset"],
                        "color": obj.get("color", "unknown"),
                    }

                # Update max visibility for each object
                max_visibility_map[key] = max(
                    max_visibility_map[key], obj["pct_pixels"]
                )

    # Restrict metadata to trajectory-salient object instances.
    salient_metadata_objs = []
    salient_type_counters = Counter()
    salient_object_bbox = defaultdict(list)
    for obj in spatial_metadata["object_bbox"]:
        if obj["object_id"] in salient_objects_n_frames:
            salient_metadata_objs.append(obj)
            salient_type_counters[obj["object_type"]] += 1
            salient_object_bbox[obj["object_type"]].append(obj)

    spatial_metadata["salient_objects"] = salient_metadata_objs
    spatial_metadata["salient_type_counts"] = salient_type_counters
    spatial_metadata["salient_object_bbox"] = salient_object_bbox

    return {
        "data": data,
        "spatial_metadata": spatial_metadata,
        "salient_objects_n_frames": salient_objects_n_frames,  # number of frames in which seen
        "salient_obj_idx_map": salient_obj_idx_map,
        "synset_map": synset_map,
        "object_info": object_info,
        "max_visibility_map": max_visibility_map,
    }


def generate_spatial_qa_pairs_for_video(
    anno_jsonl_path,
    metadata_json_path,
    video_filename,
    source,
    num_questions=5,
    question_type="temporal_rel",
    verbose=False,
    seed=42,
):
    if question_type not in QA_GEN_FNS:
        raise ValueError(f"Invalid question type: {question_type}")

    if seed is not None:
        video_identity = qa_seed_video_identity(video_filename)
        random_seed = stable_seed(seed, source, video_identity, question_type)
        random.seed(random_seed)

    try:
        preprocessed_data = preprocess_salient_objects(
            anno_jsonl_path, metadata_json_path
        )
    except IneligibleVideoError as error:
        if verbose:
            print(f"Video is ineligible for QA generation: {error}")
        return []
    except Exception as error:
        raise QAProcessingError(
            f"Failed to preprocess annotations {anno_jsonl_path}: {error}"
        ) from error

    qa_pairs = []
    attempts = 0
    max_attempts = num_questions * 1  # Limit to prevent infinite loops

    qa_gen_fn = QA_GEN_FNS[question_type]

    while attempts < max_attempts and len(qa_pairs) == 0:
        attempts += 1
        try:
            qas = qa_gen_fn(preprocessed_data, num_questions)
            if not qas:
                continue
            qa_entries = [
                {
                    "idx": i,
                    "type": "v0",  # or another appropriate type
                    "filename": video_filename,
                    "source": source,
                    **qa,
                }
                for i, qa in enumerate(qas)
            ]
            qa_pairs.extend(qa_entries)
        except ValueError as error:
            if verbose:
                print(f"No eligible '{question_type}' question: {error}")
        except Exception as error:
            raise QAProcessingError(
                f"Failed to generate '{question_type}' QAs for {anno_jsonl_path}: "
                f"{error}"
            ) from error
    return qa_pairs


def generate_qa_for_dataset_slow(
    dataset_dir,
    source,
    split="val",
    num_questions_per_video=5,
    question_type="temporal_rel",
    verbose=False,
    seed=42,
    allow_partial=False,
):
    split_dir = os.path.join(dataset_dir, split)
    scene_dirs = [
        os.path.join(split_dir, scene_dir)
        for scene_dir in sorted(os.listdir(split_dir))
        if os.path.isdir(os.path.join(split_dir, scene_dir))
    ]
    if not scene_dirs:
        raise QAProcessingError(f"No scene directories found under {split_dir}")

    summary = {"qas": 0, "videos": 0, "ineligible_videos": 0, "errors": 0}
    errors = []
    for scene_path in tqdm(scene_dirs, desc="Scenes"):
        try:
            result = process_scene(
                scene_path,
                source,
                num_questions_per_video,
                question_type,
                verbose=verbose,
                seed=seed,
            )
        except Exception as error:
            errors.append(f"{scene_path}: {type(error).__name__}: {error}")
            continue
        for key in ("qas", "videos", "ineligible_videos"):
            summary[key] += result[key]

    summary["errors"] = len(errors)
    _raise_or_report_processing_errors(errors, allow_partial)
    return summary


def process_scene(
    scene_path,
    source,
    num_questions_per_video,
    question_type,
    verbose=False,
    seed=42,
):
    """
    Process a single scene directory: For each annotation file in the scene, generate QA pairs
    and save them to a JSONL file. Returns the total number of QA pairs generated in this scene.
    """
    output_prefix = f"qa_pairs_{question_type}__"

    def remove_question_type_outputs():
        for file_name in os.listdir(scene_path):
            if file_name.startswith(output_prefix) and file_name.endswith(".jsonl"):
                os.remove(os.path.join(scene_path, file_name))

    metadata_json_path = os.path.join(scene_path, METADATA_FNAME)
    scene_dir = os.path.basename(scene_path)
    scene_total = 0
    videos_processed = 0
    ineligible_videos = 0
    pending_outputs = []
    try:
        if not os.path.exists(metadata_json_path):
            raise QAProcessingError(
                f"Metadata file {metadata_json_path} not found for scene {scene_path}"
            )

        scene_files = sorted(os.listdir(scene_path))
        annotation_indices = {
            file_name[len("offline_annos__") : -len(".jsonl")]
            for file_name in scene_files
            if file_name.startswith("offline_annos__") and file_name.endswith(".jsonl")
        }
        try:
            video_filenames = index_rgb_video_filenames(scene_files)
        except ValueError as error:
            raise QAProcessingError(
                f"Ambiguous RGB videos in scene {scene_path}: {error}"
            ) from error
        video_indices = set(video_filenames)
        if annotation_indices != video_indices:
            missing_video_indices = sorted(annotation_indices - video_indices)
            missing_annotation_indices = sorted(video_indices - annotation_indices)
            raise QAProcessingError(
                f"Annotation/video index mismatch in scene {scene_path}: "
                f"missing videos for indices {missing_video_indices}; "
                f"missing annotations for indices {missing_annotation_indices}"
            )

        for idx_part in sorted(annotation_indices):
            file_name = f"offline_annos__{idx_part}.jsonl"
            video_filename = video_filenames[idx_part]

            video_relpath = os.path.join(scene_dir, video_filename)
            anno_jsonl_path = os.path.join(scene_path, file_name)
            qa_pairs = generate_spatial_qa_pairs_for_video(
                anno_jsonl_path,
                metadata_json_path,
                video_relpath,
                source,
                num_questions=num_questions_per_video,
                question_type=question_type,
                verbose=verbose,
                seed=seed,
            )

            videos_processed += 1
            scene_total += len(qa_pairs)
            if not qa_pairs:
                ineligible_videos += 1

            qa_jsonl_filename = f"{output_prefix}{idx_part}.jsonl"
            pending_outputs.append(
                (
                    os.path.join(scene_path, qa_jsonl_filename),
                    qa_pairs,
                    video_relpath,
                )
            )

        # Replace the scene's outputs only after every video has generated
        # successfully. This also removes files for videos that no longer exist.
        remove_question_type_outputs()
        for qa_jsonl_path, qa_pairs, video_relpath in pending_outputs:
            tmp_path = f"{qa_jsonl_path}.tmp-{os.getpid()}"
            try:
                with open(tmp_path, "w") as qa_file:
                    for qa_entry in qa_pairs:
                        qa_file.write(json.dumps(qa_entry) + "\n")
                os.replace(tmp_path, qa_jsonl_path)
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            if verbose:
                print(
                    f"Generated {len(qa_pairs)} QA pairs for video {video_relpath}, saved to {qa_jsonl_path}"
                )
    except Exception:
        # A failed scene must never leave old or partially refreshed QA files
        # that a later combine stage could mistake for current output.
        remove_question_type_outputs()
        raise

    return {
        "qas": scene_total,
        "videos": videos_processed,
        "ineligible_videos": ineligible_videos,
    }


def _raise_or_report_processing_errors(errors, allow_partial):
    if not errors:
        return

    details = "\n".join(f"  - {error}" for error in sorted(errors))
    message = f"QA generation failed for {len(errors)} scene(s):\n{details}"
    if allow_partial:
        print(f"WARNING: {message}")
        return
    raise QAProcessingError(message)


def generate_qa_for_dataset(
    dataset_dir,
    source,
    split="val",
    num_questions_per_video=5,
    question_type="temporal_rel",
    num_workers=16,
    verbose=False,
    seed=42,
    allow_partial=False,
):
    """
    Generate QA pairs for the dataset in parallel over scene directories.
    After processing, print the total number of QA pairs generated for the specified question type.
    """
    split_dir = os.path.join(dataset_dir, split)
    scene_dirs = [
        os.path.join(split_dir, scene_dir)
        for scene_dir in sorted(os.listdir(split_dir))
        if os.path.isdir(os.path.join(split_dir, scene_dir))
    ]

    if not scene_dirs:
        raise QAProcessingError(f"No scene directories found under {split_dir}")

    print(
        f"Processing {len(scene_dirs)} scenes in parallel using {num_workers} workers..."
    )

    summary = {"qas": 0, "videos": 0, "ineligible_videos": 0, "errors": 0}
    errors = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(
                process_scene,
                scene,
                source,
                num_questions_per_video,
                question_type,
                verbose,
                seed,
            ): scene
            for scene in scene_dirs
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Scenes"):
            scene = futures[future]
            try:
                result = future.result()
                for key in ("qas", "videos", "ineligible_videos"):
                    summary[key] += result[key]
            except Exception as error:
                errors.append(f"{scene}: {type(error).__name__}: {error}")

    print(
        f"Generated {summary['qas']} '{question_type}' QA pairs from "
        f"{summary['videos']} videos ({summary['ineligible_videos']} ineligible)."
    )
    summary["errors"] = len(errors)
    _raise_or_report_processing_errors(errors, allow_partial)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate QA pairs for a video dataset."
    )
    parser.add_argument(
        "--dataset_dir", type=str, required=True, help="Path to the dataset directory."
    )
    parser.add_argument(
        "--split", type=str, default="val", help="The train, val, or test split."
    )
    parser.add_argument(
        "--source", type=str, required=True, help="Source identifier for the dataset."
    )
    parser.add_argument(
        "--num_questions_per_video",
        type=int,
        default=5,
        help="Number of questions to generate per video.",
    )
    parser.add_argument(
        "--question_type",
        type=str,
        choices=QA_GEN_FNS.keys(),
        default="temporal_rel",
        help="Type of questions to generate (temporal_rel, factual, or None).",
    )
    parser.add_argument(
        "--num_workers", type=int, default=16, help="Number of parallel workers to use."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--allow_partial",
        action="store_true",
        help="Keep successful scene outputs when another scene fails (default: fail).",
    )
    parser.add_argument("--verbose", action="store_true", help="Print verbose output.")

    args = parser.parse_args()
    print(f"Parsed arguments: {args}")

    generate_qa_for_dataset(
        dataset_dir=args.dataset_dir,
        source=args.source,
        split=args.split,
        num_questions_per_video=args.num_questions_per_video,
        question_type=args.question_type,
        num_workers=args.num_workers,
        verbose=args.verbose,
        seed=args.seed,
        allow_partial=args.allow_partial,
    )

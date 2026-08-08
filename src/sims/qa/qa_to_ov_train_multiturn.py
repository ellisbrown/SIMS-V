import hashlib
import json
import os
import random
import re
from collections import defaultdict

from tqdm import tqdm

from sims.qa.generators.common import stable_seed
from sims.video_paths import (
    VIDEO_VERSION_TO_STEM,
    qa_seed_video_identity,
    video_path_for_version,
)

VERS_TO_VIDFN = VIDEO_VERSION_TO_STEM


def _video_path_for_version(filename, video_version):
    return video_path_for_version(filename, video_version)


def group_qas_for_multiturn(
    all_qas,
    max_qa_per_convo=1,
    group_by_task=False,
    shuffle_within_video=True,
    seed=42,
):
    """
    Groups QA items into QA chunks.

    Args:
        all_qas (list): List of QA dicts (each from your combined JSONL).
        max_qa_per_convo (int): Number of QA pairs per chunk.
        group_by_task (bool): If True, keep tasks separate; if False, mix tasks (per video).
        shuffle_within_video (bool): If True, randomize order of QAs within each group.

    Returns:
        grouped_chunks (dict):
          A dict of { task_name : list_of_qa_chunks }
          where each list_of_qa_chunks is a list of "chunks", and each chunk is a list of QA dicts.
    """

    # Helper to define grouping keys.
    def get_video_id(qa):
        # e.g. "val/000030/rgb__0.mp4"
        return qa["filename"]

    grouped_chunks = defaultdict(list)

    # Step 1: Collect QAs by (video_id, possibly task).
    group_dict = defaultdict(list)

    for qa in tqdm(all_qas, desc="Collecting QAs by group"):
        video_id = get_video_id(qa)
        task_label = qa.get("task", "UNK")

        if group_by_task:
            group_key = (video_id, task_label)
        else:
            # We'll just put "MIXED" as the task, ignoring the actual QA's task.
            group_key = (video_id, "MIXED")
        group_dict[group_key].append(qa)

    # Step 2: For each (video_id, task_label) group, shuffle + chunk
    for (video_id, task_label), qas in tqdm(
        sorted(group_dict.items()), desc="Chunking QAs"
    ):
        if shuffle_within_video:
            seed_video_id = qa_seed_video_identity(video_id)
            random.Random(
                stable_seed(seed, seed_video_id, task_label, "format")
            ).shuffle(qas)

        # Slice them into chunks
        for start_idx in range(0, len(qas), max_qa_per_convo):
            chunk = qas[start_idx : start_idx + max_qa_per_convo]
            # If chunk is empty or partial, skip or keep it as is
            if not chunk:
                continue
            # We'll store these chunk-lists under the final "task_label" key
            # (or "MIXED" if group_by_task=False).
            grouped_chunks[task_label].append(chunk)

    return grouped_chunks


def create_multiturn_jsonl(
    dataset_dir,
    input_filename,
    output_subdir,
    output_filename_base,
    split="val",
    question_mode="mc",
    max_qa_per_convo=1,
    group_by_task=False,
    video_version="rgb",
    pre_prompt="",
    post_prompt="",
    seed=42,
    question_types=None,
    allow_partial=False,
):
    """
    Reads the combined QA JSONL, filters by MC or OE, groups into multi-turn convos,
    and writes out JSONL in a format suitable for training.

    Args:
        dataset_dir (str): Where the input and output JSONL live.
        input_filename (str): e.g. 'combined_qa_pairs.jsonl'.
        output_subdir (str): e.g. 'qas'.
        output_filename_base (str): optional prefix for formatted files.
        split (str): 'train', 'val', or 'test'.
        question_mode (str): 'mc' or 'oe'.
        max_qa_per_convo (int): # of QA pairs per conversation.
        group_by_task (bool): If True, separate files by task; else one file with tasks mixed.
        video_version (str): Which version of the video to use (e.g. 'rgb', 'depth', 'edge', ...).
        allow_partial (bool): Exclude missing videos and write missing-record
            manifests instead of failing.
    """
    if question_mode not in ["mc", "mc_direct", "oe"]:
        raise ValueError("Invalid question_mode. Use 'mc', 'mc_direct', or 'oe'.")

    input_path = os.path.join(dataset_dir, split, input_filename)

    # Step 1: Load all QAs
    all_qas = []
    with open(input_path, "r") as infile:
        for line in infile:
            qa = json.loads(line.strip())
            if question_types is not None and qa.get("task") not in question_types:
                continue
            all_qas.append(qa)

    if not all_qas:
        requested = "all question types" if question_types is None else question_types
        raise ValueError(f"No QA records found for {requested} in {input_path}")

    missing_videos = sorted(
        {
            _video_path_for_version(qa["filename"], video_version)
            for qa in all_qas
            if not os.path.exists(
                os.path.join(
                    dataset_dir,
                    _video_path_for_version(qa["filename"], video_version),
                )
            )
        }
    )
    if missing_videos and not allow_partial:
        preview = ", ".join(missing_videos[:10])
        if len(missing_videos) > 10:
            preview += f", ... and {len(missing_videos) - 10} more"
        raise FileNotFoundError(
            f"Missing {len(missing_videos)} selected video file(s) under "
            f"{dataset_dir}: {preview}. Use --allow_partial to exclude records "
            "whose videos are missing."
        )

    # Step 2: Group them by (video_id, task) or (video_id, 'MIXED'), then chunk
    grouped_chunks = group_qas_for_multiturn(
        all_qas=all_qas,
        max_qa_per_convo=max_qa_per_convo,
        group_by_task=group_by_task,
        shuffle_within_video=True,
        seed=seed,
    )

    subdir = os.path.join(dataset_dir, output_subdir, split, video_version)
    os.makedirs(subdir, exist_ok=True)

    # Files matching this formatter's naming scheme are derived artifacts.
    # Remove them before rebuilding so a rerun with fewer question types cannot
    # leave stale task files alongside the newly requested corpus.
    prefix = "" if output_filename_base == "" else f"{output_filename_base}_"
    formatter_output = re.compile(
        rf"^(?:missing_)?{re.escape(prefix)}mt\d+_.+_{re.escape(question_mode)}\.jsonl$"
    )
    for filename in sorted(os.listdir(subdir)):
        path = os.path.join(subdir, filename)
        if os.path.isfile(path) and formatter_output.fullmatch(filename):
            os.remove(path)

    # If group_by_task=True, we will create multiple output files (one per task).
    # If group_by_task=False, we create just one file with all tasks "MIXED".
    total_convs = 0
    total_missing = 0
    total_chunks = 0
    missing_videos = set()

    for task_label, chunks in tqdm(
        sorted(grouped_chunks.items()), desc="Writing multi-turn JSONLs"
    ):
        # e.g. "descriptive_binary", "temporal_rel", ...
        # Build multi-turn records for these chunks.
        final_convos, missing_convos = build_multi_turn_convos(
            chunks, question_mode, dataset_dir, video_version, pre_prompt, post_prompt
        )

        total_convs += len(final_convos)
        total_missing += len(missing_convos)
        total_chunks += len(chunks)

        # Example output filename: mt1_descriptive_binary_mc.jsonl
        out_filename = (
            f"{prefix}mt{max_qa_per_convo}_{task_label}_{question_mode}.jsonl"
        )
        output_path = os.path.join(subdir, out_filename)

        write_conversations_to_jsonl(final_convos, output_path)
        print(
            f"Wrote {len(final_convos)} conversations for task={task_label} to {output_path}"
        )

        # write missing convos to a separate file
        missing_output_path = os.path.join(subdir, f"missing_{out_filename}")
        if len(missing_convos) > 0:
            write_conversations_to_jsonl(missing_convos, missing_output_path)
            print(
                f"Wrote {len(missing_convos)} missing conversations for task={task_label} to {missing_output_path}"
            )
            missing_videos.update({c["video"] for c in missing_convos})
        else:
            print(f"No missing conversations for task={task_label}")

    print(f"Total conversations: {total_convs}, total chunks: {total_chunks}")
    print(f"Total missing video convos (excluded): {total_missing}")
    if total_missing > 0:
        print(f"Distinct missing videos: {len(missing_videos)}")
        print("First 10 missing videos:")
        for path in sorted(missing_videos)[:10]:
            print(f"  - {path}")
        if len(missing_videos) > 10:
            print(f"  ... and {len(missing_videos) - 10} more")


def build_multi_turn_convos(
    chunks,
    question_mode,
    dataset_dir,
    video_version="rgb",
    pre_prompt="",
    post_prompt="",
):
    """
    Convert chunked QAs into multi-turn conversation dicts
    in the LLaVA / Open-Vision style.
    Each chunk corresponds to one conversation.

    Args:
        chunks (list): A list of chunk-lists, each chunk-list is a list of QAs
                       that share the same video (and possibly the same task).
        question_mode (str): 'mc' or 'oe'
        dataset_dir (str): Root dataset directory (for resolving video paths).
        video_version (str): Which version of the video to use (e.g. 'rgb', 'depth', 'edge', ...).
        pre_prompt (str): Instruction to add before each question.
        post_prompt (str): Instruction to add after each question.

    Returns:
        tuple: (final_convos, missing_convos)
            final_convos (list): List of conversation dicts
            missing_convos (list): List of conversation dicts with missing videos
    """
    final_convos = []
    missing_convos = []

    for chunk in chunks:
        # Since we grouped by (video_id, task), all QAs in a chunk should share the same video + source.
        # Let's do a quick consistency check.
        dataset_versions = {qa_item["source"] for qa_item in chunk}
        filenames = {qa_item["filename"] for qa_item in chunk}
        assert len(dataset_versions) == 1, (
            f"Inconsistent source in chunk: {dataset_versions}"
        )
        assert len(filenames) == 1, f"Inconsistent filename in chunk: {filenames}"

        dataset_version = next(iter(dataset_versions))
        filename = next(iter(filenames))
        video_path = filename

        # A stable content ID remains unique when task groups are written to
        # separate files; a per-file counter would collide across those files.
        source = f"sims_{dataset_version}"

        # We'll build a conversation with alternating 'human' and 'gpt' turns.
        conversation_turns = []
        for qa_item in chunk:
            task_label = qa_item.get("task", "UNK")
            question_str = ""
            if pre_prompt:
                question_str += pre_prompt + "\n"
            if question_mode == "mc":
                question_str += qa_item["mc_question"]  # typical MC question text
                answer = f"{qa_item['mc_answer']}. {qa_item['gt_answer']}"
            elif question_mode == "mc_direct":
                question_str += qa_item["mc_question"]  # typical MC question text
                answer = f"{qa_item['mc_answer']}"
            else:
                # Open-ended question
                question_str += qa_item["question"]
                answer = str(qa_item["gt_answer"])
            if post_prompt:
                question_str += "\n" + post_prompt

            conversation_turns.append(
                {"from": "human", "value": question_str, "task": task_label}
            )
            conversation_turns.append({"from": "gpt", "value": answer})

        # convert video_filename to desired version
        video_path = _video_path_for_version(video_path, video_version)

        identity_payload = {
            "source": dataset_version,
            "video": video_path,
            "video_version": video_version,
            "question_mode": question_mode,
            # The combined-corpus index changes when question types are
            # filtered, so it is not part of the QA's stable identity.
            "qas": [
                {key: value for key, value in qa.items() if key != "idx"}
                for qa in chunk
            ],
        }
        identity = json.dumps(
            identity_payload, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        conv_id = f"{source}__{hashlib.sha256(identity).hexdigest()}"

        # Construct final conversation record
        convo_record = {
            "id": conv_id,
            "conversations": conversation_turns,
            "type": question_mode,
            "data_source": f"sims_{dataset_version}",
            "video": video_path,
        }
        # Check if video exists - just record if missing instead of failing
        if not os.path.exists(os.path.join(dataset_dir, video_path)):
            missing_convos.append(convo_record)
        else:
            final_convos.append(convo_record)

    return final_convos, missing_convos


def write_conversations_to_jsonl(conversations, output_path):
    """
    Write a list of conversation dicts to JSONL.
    """
    with open(output_path, "w") as outfile:
        for record in conversations:
            outfile.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Transform QA pairs into multi-turn conversation format."
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Directory of combined QA JSONL and output.",
    )
    parser.add_argument(
        "--input_filename",
        type=str,
        default="combined_qa_pairs.jsonl",
        help="Input QA JSONL.",
    )
    parser.add_argument(
        "--output_subdir",
        type=str,
        default="qas",
        help="Subdirectory to write the output JSONL.",
    )
    parser.add_argument(
        "--output_filename_base",
        type=str,
        default="",
        help="Base name for the output JSONL (we append task or 'mixed').",
    )
    parser.add_argument(
        "--split", type=str, default="val", help="The train, val, or test split."
    )
    parser.add_argument(
        "--question_type",
        choices=["mc", "mc_direct", "oe"],
        default="mc",
        help="Which type of QAs to keep: 'mc' or 'oe'.",
    )
    parser.add_argument(
        "--max_qa_per_convo",
        type=int,
        default=1,
        help="Number of QAs per multi-turn conversation.",
    )
    parser.add_argument(
        "--group_by_task",
        action="store_true",
        help="If True, separate QAs by their 'task' and write multiple files (one per task).",
    )
    parser.add_argument(
        "--video_version",
        type=str,
        default="rgb",
        choices=VERS_TO_VIDFN.keys(),
        help="Which version of the video to use.",
    )
    parser.add_argument(
        "--pre_prompt",
        type=str,
        default="",
        help="Pre-prompt instruction to add to questions.",
    )
    parser.add_argument(
        "--post_prompt",
        type=str,
        default="",
        help="Post-prompt instruction to add to questions.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--question_types",
        nargs="+",
        default=None,
        help="Only format these exact question types (default: all).",
    )
    parser.add_argument(
        "--allow_partial",
        action="store_true",
        help="Exclude records with missing videos instead of failing.",
    )
    args = parser.parse_args()

    create_multiturn_jsonl(
        dataset_dir=args.dataset_dir,
        input_filename=args.input_filename,
        output_subdir=args.output_subdir,
        output_filename_base=args.output_filename_base,
        split=args.split,
        question_mode=args.question_type,
        max_qa_per_convo=args.max_qa_per_convo,
        group_by_task=args.group_by_task,
        video_version=args.video_version,
        pre_prompt=args.pre_prompt,
        post_prompt=args.post_prompt,
        seed=args.seed,
        question_types=args.question_types,
        allow_partial=args.allow_partial,
    )

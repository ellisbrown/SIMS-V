import os
import json
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from tqdm import tqdm


def process_file_chunk(qa_file_paths, split):
    """
    Processes a chunk of QA JSONL files.

    Args:
        qa_file_paths (list): List of QA JSONL file paths.
        split (str): The dataset split (e.g., "train", "val", "test").

    Returns:
        list: A list of QA entry dictionaries from all files in the chunk,
              with the 'filename' field updated.
    """
    qa_entries = []
    for qa_file_path in qa_file_paths:
        with open(qa_file_path, "r") as infile:
            for line_number, line in enumerate(infile, start=1):
                line = line.strip()
                if line:
                    try:
                        qa_entry = json.loads(line)
                    except json.JSONDecodeError as error:
                        raise ValueError(
                            f"Invalid JSON in {qa_file_path}:{line_number}"
                        ) from error
                    if "filename" not in qa_entry:
                        raise ValueError(
                            f"Missing 'filename' in {qa_file_path}:{line_number}"
                        )
                    # Update the filename to include the split as a prefix
                    qa_entry["filename"] = os.path.join(split, qa_entry["filename"])
                    qa_entries.append(qa_entry)
    return qa_entries


def chunks(lst, n):
    """
    Yield successive n-sized chunks from lst.

    Args:
        lst (list): The list to split.
        n (int): Chunk size.

    Yields:
        list: Chunks of the original list.
    """
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def combine_qa_files(
    dataset_dir,
    split,
    output_filename,
    num_workers=16,
    chunk_size=100,
    question_types=None,
):
    """
    Combines all QA JSONL files from subdirectories into a single JSONL file.
    Renumbers the 'idx' field.

    This version processes files in parallel using chunking.

    Args:
        dataset_dir (str): Path to the dataset directory containing subdirectories with QA files.
        split (str): The train, val, or test split.
        output_filename (str): Name of the output combined JSONL file. Saved in the dataset_dir root.
        num_workers (int): Number of parallel workers to use.
        chunk_size (int): Number of files to process per task.
        question_types (collection[str] | None): If provided, include only
            per-video files for these exact question types.
    """
    output_file = os.path.join(dataset_dir, split, output_filename)
    print(
        f"\nCombining QA pairs from '{split}' split into a single JSONL file using {num_workers} workers..."
    )

    # Gather all QA file paths from the split directory.
    print(f"Scanning '{split}' split directory for QA JSONL files...")
    split_dir = os.path.join(dataset_dir, split)
    qa_file_paths = []
    requested_prefixes = None
    if question_types is not None:
        requested_prefixes = tuple(
            f"qa_pairs_{question_type}__" for question_type in question_types
        )
        if not requested_prefixes:
            raise ValueError("question_types must not be empty")

    for root, dirs, files in tqdm(os.walk(split_dir), desc="Scanning QA files"):
        dirs.sort()
        for file in sorted(files):
            is_qa_file = file.startswith("qa_pairs_") and file.endswith(".jsonl")
            is_requested = requested_prefixes is None or file.startswith(
                requested_prefixes
            )
            if is_qa_file and is_requested:
                qa_file_paths.append(os.path.join(root, file))

    qa_file_paths.sort()

    print(f"Found {len(qa_file_paths)} QA JSONL files.")
    if not qa_file_paths:
        raise ValueError(f"No QA JSONL files found under {split_dir}")

    # Create chunks of file paths.
    qa_chunks = list(chunks(qa_file_paths, chunk_size))
    print(
        f"Processing {len(qa_chunks)} chunks (each with up to {chunk_size} files) in parallel..."
    )

    all_entries = []
    if num_workers == 1:
        results = [process_file_chunk(chunk, split) for chunk in qa_chunks]
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Use executor.map to process each chunk; pass the same split value.
            results = list(
                tqdm(
                    executor.map(process_file_chunk, qa_chunks, repeat(split)),
                    total=len(qa_chunks),
                    desc="Processing QA file chunks",
                )
            )

    # Flatten the results (each result is a list of QA entries).
    for res in results:
        all_entries.extend(res)

    if not all_entries:
        raise ValueError(f"QA files under {split_dir} contained no records")

    print(
        f"\nFinished processing all QA JSONL files. Saving combined QA pairs to {output_file}..."
    )

    print("Sorting and renumbering QA entries...")
    print(f"Total QA pairs before sorting: {len(all_entries)}")

    print(f"Keys: {all_entries[0].keys()}")
    print(f"First entry: {all_entries[0]}")

    # sort them deterministically before renumbering
    all_entries.sort(key=lambda x: x.get("id", 0))  # house-level questions have no id
    all_entries.sort(key=lambda x: x["type"])
    all_entries.sort(key=lambda x: x["task"])
    all_entries.sort(key=lambda x: x["filename"])

    # Renumber the 'idx' field sequentially.
    for idx, entry in enumerate(all_entries):
        entry["idx"] = idx

    # Write all combined QA entries to the output file.
    with open(output_file, "w") as outfile:
        for entry in all_entries:
            outfile.write(json.dumps(entry) + "\n")

    print(f"\nCombined QA pairs saved to {output_file}")
    print(f"Total QA pairs combined: {len(all_entries)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Combine QA JSONL files into a single JSONL file."
    )
    parser.add_argument(
        "--dataset_dir", type=str, required=True, help="Path to the dataset directory."
    )
    parser.add_argument(
        "--split", type=str, default="val", help="The train, val, or test split."
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        default="combined_qa_pairs.jsonl",
        help="Name of the output combined JSONL file.",
    )
    parser.add_argument(
        "--num_workers", type=int, default=16, help="Number of parallel workers to use."
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=100,
        help="Number of files to process per chunk.",
    )
    parser.add_argument(
        "--question_types",
        nargs="+",
        default=None,
        help="Only combine these exact question types (default: all).",
    )
    args = parser.parse_args()

    combine_qa_files(
        args.dataset_dir,
        args.split,
        args.output_filename,
        num_workers=args.num_workers,
        chunk_size=args.chunk_size,
        question_types=args.question_types,
    )

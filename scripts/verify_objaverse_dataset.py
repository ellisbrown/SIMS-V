#!/usr/bin/env python3
"""
Verify Objaverse dataset configuration and structure.

This script checks that the Objaverse SIMS dataset is properly configured
and all required files are present.

Exit codes:
    0 - All checks passed
    1 - One or more checks failed
"""

import argparse
import os
import sys
from pathlib import Path


DATASET_REVISION = "9087d8d6df551c3ad0af85b1e2b24fa6f654ae7d"


class Colors:
    """ANSI color codes for terminal output."""

    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BLUE = "\033[94m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


def print_check(message, status, details=None):
    """Print a check result with color coding."""
    if status == "pass":
        symbol = f"{Colors.GREEN}✓{Colors.RESET}"
    elif status == "warn":
        symbol = f"{Colors.YELLOW}⚠{Colors.RESET}"
    else:
        symbol = f"{Colors.RED}✗{Colors.RESET}"

    print(f"{symbol} {message}")
    if details:
        print(f"  {Colors.BLUE}{details}{Colors.RESET}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Verify the downloaded ProcTHOR-Objaverse data bundle."
    )
    parser.add_argument(
        "--objaverse-dir",
        help=(
            "Dataset directory. Defaults to OBJAVERSE_DATA_DIR, then ./objaverse_sims."
        ),
    )
    return parser.parse_args(argv)


def main(argv=None):
    """Run all verification checks."""
    args = parse_args(argv)
    print(f"\n{Colors.BOLD}Objaverse Dataset Verification{Colors.RESET}")
    print("=" * 60)

    all_passed = True

    # Check 1: Path resolution
    print(f"\n{Colors.BOLD}[1] Dataset Location{Colors.RESET}")
    objaverse_data_dir = os.environ.get("OBJAVERSE_DATA_DIR")

    if args.objaverse_dir:
        base_dir = Path(args.objaverse_dir)
        print_check("Using --objaverse-dir", "pass", f"Path: {base_dir}")
    elif objaverse_data_dir:
        print_check("OBJAVERSE_DATA_DIR is set", "pass", f"Path: {objaverse_data_dir}")
        base_dir = Path(objaverse_data_dir)
    else:
        print_check(
            "OBJAVERSE_DATA_DIR not set, using ./objaverse_sims",
            "warn",
            f"Path: {Path.cwd() / 'objaverse_sims'}",
        )
        base_dir = Path.cwd() / "objaverse_sims"

    # Resolve to absolute path
    base_dir = base_dir.resolve()

    # Check 2: Base directory exists
    print(f"\n{Colors.BOLD}[2] Base Directory{Colors.RESET}")
    if base_dir.exists():
        print_check("Base directory exists", "pass", str(base_dir))
    else:
        print_check("Base directory does NOT exist", "fail", str(base_dir))
        all_passed = False
        print(f"\n{Colors.RED}Cannot proceed with further checks.{Colors.RESET}")
        print("Pass --objaverse-dir for another location or follow the download")
        print("instructions in docs/getting-started.md.")
        return 1

    # Check 3: Subdirectories
    print(f"\n{Colors.BOLD}[3] Required Subdirectories{Colors.RESET}")
    subdirs = {
        "processed": base_dir / "processed",
        "houses": base_dir / "houses",
        "procthor_databases": base_dir / "procthor_databases",
    }

    for name, path in subdirs.items():
        if path.exists() and path.is_dir():
            # Count items for context
            if name == "processed":
                item_count = len([d for d in path.iterdir() if d.is_dir()])
                print_check(
                    f"{name}/ exists", "pass", f"{item_count} object directories found"
                )
            else:
                print_check(f"{name}/ exists", "pass", str(path))
        else:
            print_check(f"{name}/ directory NOT found", "fail", f"Expected: {path}")
            all_passed = False

    # Check 4: Critical house files
    print(f"\n{Colors.BOLD}[4] House Layout Files{Colors.RESET}")
    houses_dir = subdirs["houses"]
    house_files = {
        "train": houses_dir / "train.jsonl.gz",
        "test": houses_dir / "test.jsonl.gz",
        "val": houses_dir / "val.jsonl.gz",
    }

    for split, filepath in house_files.items():
        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024 * 1024)
            print_check(f"{split}.jsonl.gz exists", "pass", f"{size_mb:.1f} MB")
        else:
            print_check(f"{split}.jsonl.gz NOT found", "fail", str(filepath))
            all_passed = False

    # Check 5: Database files
    print(f"\n{Colors.BOLD}[5] ProcTHOR Database Files{Colors.RESET}")
    databases_dir = subdirs["procthor_databases"]
    database_files = [
        "asset-database.json",
        "material-database.json",
        "placement-annotations.json",
        "receptacles.json",
    ]

    for filename in database_files:
        filepath = databases_dir / filename
        if filepath.exists():
            size_kb = filepath.stat().st_size / 1024
            print_check(f"{filename} exists", "pass", f"{size_kb:.1f} KB")
        else:
            print_check(f"{filename} NOT found", "fail", str(filepath))
            all_passed = False

    # Check 6: Sample processed objects
    print(f"\n{Colors.BOLD}[6] Processed Objects{Colors.RESET}")
    processed_dir = subdirs["processed"]
    if processed_dir.exists():
        # Find first object directory
        object_dirs = [d for d in processed_dir.iterdir() if d.is_dir()]
        if object_dirs:
            sample_dir = object_dirs[0]
            required_files = [
                f"{sample_dir.name}.pkl.gz",
                "albedo.jpg",
                "normal.jpg",
                "emission.jpg",
                "thor_metadata.json",
            ]

            sample_valid = True
            for filename in required_files:
                if not (sample_dir / filename).exists():
                    sample_valid = False
                    break

            if sample_valid:
                print_check(
                    "Sample object structure valid",
                    "pass",
                    f"Checked: {sample_dir.name}",
                )
            else:
                print_check(
                    "Sample object missing required files",
                    "warn",
                    f"Checked: {sample_dir.name}",
                )
        else:
            print_check(
                "No object directories found in processed/", "fail", str(processed_dir)
            )
            all_passed = False

    # Check 7: WordNet corpus used by task and QA utilities
    print(f"\n{Colors.BOLD}[7] NLTK Data{Colors.RESET}")
    try:
        from nltk.corpus import wordnet2022 as wn

        wn.synset("physical_entity.n.01")
        print_check("wordnet2022 corpus is available", "pass")
    except LookupError:
        print_check(
            "wordnet2022 corpus is not installed",
            "fail",
            "Run: uv run python -m nltk.downloader wordnet wordnet2022",
        )
        all_passed = False

    # Summary
    print("\n" + "=" * 60)
    if all_passed:
        print(f"{Colors.GREEN}{Colors.BOLD}✓ All checks passed!{Colors.RESET}")
        print("\nDataset is correctly configured and ready to use.")
        print(f"\nResolved dataset: {base_dir}")
        return 0
    else:
        print(f"{Colors.RED}{Colors.BOLD}✗ Some checks failed{Colors.RESET}")
        print("\nPlease ensure the dataset is properly downloaded and extracted.")
        print("\nTo download the pinned dataset snapshot:")
        print(
            "  uvx --from huggingface-hub hf download ellisbrown/objaverse_sims "
            f"--repo-type dataset --revision {DATASET_REVISION} "
            f"--local-dir {base_dir}"
        )
        print("  Then follow the extraction commands in docs/getting-started.md.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

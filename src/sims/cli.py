"""Public command-line interface for SIMS-V generation."""

import argparse
import sys


COMMANDS = {
    "generate": "Generate simulated house walkthroughs and sensor recordings.",
    "qa": "Generate metadata, video questions, and training JSONL.",
}


def build_parser():
    parser = argparse.ArgumentParser(
        prog="sims-v",
        description="Generate simulated spatial-video training data.",
    )
    parser.add_argument("command", nargs="?", choices=COMMANDS)
    return parser


def _print_help(parser):
    parser.print_help()
    print("\ncommands:")
    for command, description in COMMANDS.items():
        print(f"  {command:<10} {description}")
    print("\nRun `sims-v <command> --help` for command-specific options.")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()

    if not argv or argv[0] in {"-h", "--help"}:
        _print_help(parser)
        return 0

    command = argv.pop(0)
    if command not in COMMANDS:
        parser.error(f"unknown command: {command}")

    if command == "generate":
        from sims.data_generation.datagen_scripts.offline_video_datagen_mpqueue import (
            main as generate_main,
        )

        return generate_main(argv, prog="sims-v generate")

    from sims.pipeline import main as qa_main

    return qa_main(argv, prog="sims-v qa")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run pluggable LLM-judge metrics over canonical agent data."""

from __future__ import annotations

import argparse
from pathlib import Path

from core.runner import run_from_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run LLM-judge metrics over canonical JSONL/JSONL.GZ data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to judge YAML/JSON config.",
    )
    parser.add_argument(
        "--input",
        nargs="+",
        default=None,
        help=(
            "Override input.paths from config. Accepts files, directories, or glob "
            "patterns. Directories are searched recursively for JSON/JSONL files."
        ),
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Override input.max_samples from config.",
    )
    parser.add_argument(
        "--balanced-sample",
        action="store_true",
        help=(
            "When max_samples is set, sample approximately equal numbers of "
            "records from each discovered input file."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used by --balanced-sample.",
    )
    args = parser.parse_args()
    run_from_config(
        Path(args.config),
        input_paths=args.input,
        max_samples=args.max_samples,
        balanced_sample=args.balanced_sample,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Summarize 0/1/2 score distributions for metrics in one judge run."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


SCRIPT_PATH = Path(__file__).resolve()
TOOLS_ROOT = SCRIPT_PATH.parent
JUDGE_ROOT = TOOLS_ROOT.parent
OUTPUTS_ROOT = JUDGE_ROOT / "outputs"
REPO_ROOT = JUDGE_ROOT.parent

sys.path.insert(0, str(JUDGE_ROOT))

from core.io import read_jsonl  # noqa: E402


SCORE_LABELS = ("0", "1", "2")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Read task_results.jsonl from one judge run and summarize 0/1/2 "
            "score distributions for each metric."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run-dir", required=True, help="Judge run directory.")
    parser.add_argument(
        "--metric",
        action="append",
        default=None,
        help="Metric name to include. May be passed more than once. Defaults to all metrics.",
    )
    parser.add_argument(
        "--task-id",
        default="sample",
        help="Only include rows with this task_id.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON summary output path.",
    )
    args = parser.parse_args(argv)

    run_dir = resolve_run_dir(args.run_dir)
    summary = summarize_metric_score_distribution(
        run_dir,
        metric_filter=set(args.metric or []),
        task_id=args.task_id,
    )
    print_summary(summary)

    if args.output:
        output_path = resolve_output_path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"[score-distribution] output: {output_path}")


def summarize_metric_score_distribution(
    run_dir: str | Path,
    *,
    metric_filter: set[str],
    task_id: str,
) -> dict[str, Any]:
    """Return per-metric counts for valid scores and invalid/error rows."""
    resolved_run_dir = resolve_run_dir(run_dir)
    result_path = resolved_run_dir / "task_results.jsonl"
    if not result_path.exists():
        raise FileNotFoundError(f"Missing task_results.jsonl: {result_path}")

    buckets: dict[str, dict[str, Any]] = defaultdict(new_metric_bucket)
    for row in read_jsonl(result_path):
        if str(row.get("task_id")) != task_id:
            continue

        raw_metric = row.get("metric")
        if raw_metric is None:
            continue
        metric = str(raw_metric)
        if metric_filter and metric not in metric_filter:
            continue

        bucket = buckets[metric]
        bucket["total_results"] += 1
        score = normalize_score(row.get("score"))
        if score is None or row.get("error") is not None:
            bucket["unscored_or_invalid"] += 1
            continue

        bucket["valid_scored_results"] += 1
        bucket["score_counts"][score] += 1

    metrics = {
        metric: finalize_metric_bucket(buckets[metric])
        for metric in sorted(buckets)
    }
    return {
        "run_dir": str(resolved_run_dir),
        "task_id": task_id,
        "metrics": metrics,
    }


def new_metric_bucket() -> dict[str, Any]:
    return {
        "total_results": 0,
        "valid_scored_results": 0,
        "score_counts": {label: 0 for label in SCORE_LABELS},
        "unscored_or_invalid": 0,
    }


def finalize_metric_bucket(bucket: dict[str, Any]) -> dict[str, Any]:
    valid_scored_results = bucket["valid_scored_results"]
    score_counts = dict(bucket["score_counts"])
    percentages = {
        label: round(score_counts[label] / valid_scored_results * 100, 2)
        if valid_scored_results
        else 0.0
        for label in SCORE_LABELS
    }
    return {
        "total_results": bucket["total_results"],
        "valid_scored_results": valid_scored_results,
        "score_counts": score_counts,
        "score_percentages": percentages,
        "unscored_or_invalid": bucket["unscored_or_invalid"],
    }


def normalize_score(value: Any) -> str | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return str(value) if value in {0, 1, 2} else None
    if isinstance(value, float):
        if not value.is_integer():
            return None
        numeric = int(value)
        return str(numeric) if numeric in {0, 1, 2} else None
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        try:
            numeric_float = float(raw)
        except ValueError:
            return None
        if not numeric_float.is_integer():
            return None
        numeric = int(numeric_float)
        return str(numeric) if numeric in {0, 1, 2} else None
    return None


def resolve_run_dir(raw: str | Path) -> Path:
    path = Path(raw)
    candidates = [
        path,
        OUTPUTS_ROOT / path,
        JUDGE_ROOT / path,
        REPO_ROOT / path,
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate.resolve()
    raise FileNotFoundError(f"Run directory not found: {raw}")


def resolve_output_path(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else (Path.cwd() / path).resolve()


def print_summary(summary: dict[str, Any]) -> None:
    print(f"[score-distribution] run_dir: {summary['run_dir']}")
    print(f"[score-distribution] task_id: {summary['task_id']}")
    for metric, bucket in summary["metrics"].items():
        print()
        print(f"[score-distribution] metric: {metric}")
        print(f"[score-distribution] total results: {bucket['total_results']}")
        for label in SCORE_LABELS:
            count = bucket["score_counts"][label]
            percentage = bucket["score_percentages"][label]
            print(f"[score-distribution] score {label}: {count} ({percentage:.2f}%)")
        print(
            "[score-distribution] "
            f"unscored_or_invalid: {bucket['unscored_or_invalid']}"
        )


if __name__ == "__main__":
    main()

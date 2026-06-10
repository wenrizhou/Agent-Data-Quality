#!/usr/bin/env python3
"""Summarize score agreements between two judge runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


SCRIPT_PATH = Path(__file__).resolve()
TOOLS_ROOT = SCRIPT_PATH.parent
JUDGE_ROOT = TOOLS_ROOT.parent
OUTPUTS_ROOT = JUDGE_ROOT / "outputs"
REPO_ROOT = JUDGE_ROOT.parent

sys.path.insert(0, str(JUDGE_ROOT))

from core.io import read_jsonl  # noqa: E402


ResultKey = tuple[str, str, str]
SCORE_LABELS = ("0", "1", "2")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare two judge runs by sample_id/metric/task_id, find rows with "
            "the same score, and count agreed 0/1/2 scores."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run-a", required=True, help="First judge run directory.")
    parser.add_argument("--run-b", required=True, help="Second judge run directory.")
    parser.add_argument(
        "--metric",
        action="append",
        default=None,
        help=(
            "Metric name to include. May be passed more than once. "
            "Defaults to all metrics common to both runs."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON summary output path.",
    )
    args = parser.parse_args(argv)

    run_a = resolve_run_dir(args.run_a)
    run_b = resolve_run_dir(args.run_b)
    summary = summarize_agreements(
        run_a,
        run_b,
        metric_filter=set(args.metric or []),
    )
    print_summary(summary)
    if args.output:
        output_path = resolve_output_path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"[agreement] output: {output_path}")


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


def summarize_agreements(
    run_a: Path,
    run_b: Path,
    *,
    metric_filter: set[str],
) -> dict[str, Any]:
    results_a = load_result_index(run_a, metric_filter=metric_filter)
    results_b = load_result_index(run_b, metric_filter=metric_filter)
    keys_a = set(results_a)
    keys_b = set(results_b)
    common_keys = sorted(keys_a & keys_b, key=lambda key: (key[1], key[2], key[0]))

    counts = {label: 0 for label in SCORE_LABELS}
    agreements = 0
    disagreements = 0
    unscored_or_invalid = 0

    for key in common_keys:
        score_a = normalize_score(results_a[key].get("score"))
        score_b = normalize_score(results_b[key].get("score"))
        if score_a is None or score_b is None:
            unscored_or_invalid += 1
            continue
        if score_a == score_b:
            agreements += 1
            counts[score_a] += 1
        else:
            disagreements += 1

    return {
        "run_a": str(run_a.resolve()),
        "run_b": str(run_b.resolve()),
        "metrics": sorted(metric_filter) if metric_filter else "all",
        "results_in_run_a": len(keys_a),
        "results_in_run_b": len(keys_b),
        "common_results": len(common_keys),
        "agreements": agreements,
        "disagreements": disagreements,
        "unscored_or_invalid": unscored_or_invalid,
        "missing_from_run_a": len(keys_b - keys_a),
        "missing_from_run_b": len(keys_a - keys_b),
        "agreed_score_counts": counts,
    }


def load_result_index(
    run_dir: Path,
    *,
    metric_filter: set[str],
) -> dict[ResultKey, dict[str, Any]]:
    result_path = run_dir / "task_results.jsonl"
    if not result_path.exists():
        raise FileNotFoundError(f"Missing task_results.jsonl: {run_dir}")

    rows = read_jsonl(result_path)
    if not rows:
        raise RuntimeError(f"No task results found in {result_path}")

    indexed: dict[ResultKey, dict[str, Any]] = {}
    for row in rows:
        key = result_key(row)
        if key is None:
            continue
        if metric_filter and key[1] not in metric_filter:
            continue
        indexed[key] = row
    return indexed


def result_key(row: dict[str, Any]) -> ResultKey | None:
    sample_id = row.get("sample_id")
    metric = row.get("metric")
    task_id = row.get("task_id")
    if sample_id is None or metric is None or task_id is None:
        return None
    return str(sample_id), str(metric), str(task_id)


def normalize_score(value: Any) -> str | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return str(value) if value in {0, 1, 2} else None
    if isinstance(value, float):
        numeric = int(value)
        return str(numeric) if value == numeric and numeric in {0, 1, 2} else None
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        try:
            numeric_float = float(raw)
        except ValueError:
            return None
        numeric = int(numeric_float)
        return str(numeric) if numeric_float == numeric and numeric in {0, 1, 2} else None
    return None


def print_summary(summary: dict[str, Any]) -> None:
    print(f"[agreement] run_a: {summary['run_a']}")
    print(f"[agreement] run_b: {summary['run_b']}")
    print(f"[agreement] metrics: {summary['metrics']}")
    print(f"[agreement] results in run_a: {summary['results_in_run_a']}")
    print(f"[agreement] results in run_b: {summary['results_in_run_b']}")
    print(f"[agreement] common results: {summary['common_results']}")
    print(f"[agreement] agreements: {summary['agreements']}")
    print(f"[agreement] disagreements: {summary['disagreements']}")
    print(f"[agreement] unscored/invalid: {summary['unscored_or_invalid']}")
    print(f"[agreement] missing from run_a: {summary['missing_from_run_a']}")
    print(f"[agreement] missing from run_b: {summary['missing_from_run_b']}")
    for label in SCORE_LABELS:
        print(f"[agreement] agreed score {label}: {summary['agreed_score_counts'][label]}")


if __name__ == "__main__":
    main()

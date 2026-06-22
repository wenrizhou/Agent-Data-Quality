#!/usr/bin/env python3
"""Export one-row-per-sample metric results from a judge run."""

from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable


SCRIPT_PATH = Path(__file__).resolve()
TOOLS_ROOT = SCRIPT_PATH.parent
JUDGE_ROOT = TOOLS_ROOT.parent
OUTPUTS_ROOT = JUDGE_ROOT / "outputs"
REPO_ROOT = JUDGE_ROOT.parent

sys.path.insert(0, str(JUDGE_ROOT))

from core.io import load_config, read_jsonl, write_jsonl  # noqa: E402


DEFAULT_EXPECTED_METRICS = [
    "specificity_en",
    "parameter_alignment_en",
    "tool_selection_en",
    "user_requirement_fulfillment_en",
    "final_response_evidence_support_en",
    "minimality_en",
]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Read task_results.jsonl from a judge run, group rows by sample_id, "
            "and write sample_metric_results.jsonl with a derived data_quality "
            "score equal to the minimum source metric score."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run-dir", required=True, help="Judge run directory.")
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSONL path. Defaults to <run-dir>/sample_metric_results.jsonl.",
    )
    parser.add_argument(
        "--metric",
        action="append",
        default=None,
        help=(
            "Expected source metric. May be repeated. Defaults to metrics from "
            "run_config if available, otherwise the English all-metrics set."
        ),
    )
    parser.add_argument(
        "--task-id",
        default="sample",
        help="Only aggregate task_results rows with this task_id.",
    )
    args = parser.parse_args(argv)

    run_dir = resolve_run_dir(args.run_dir)
    expected_metrics = args.metric or infer_expected_metrics(run_dir)
    output_path = (
        resolve_output_path(args.output)
        if args.output
        else run_dir / "sample_metric_results.jsonl"
    )
    written = export_sample_metric_results(
        run_dir,
        output_path=output_path,
        expected_metrics=expected_metrics,
        task_id=args.task_id,
    )
    print(f"[export] run_dir: {run_dir}")
    print(f"[export] output: {written}")


def export_sample_metric_results(
    run_dir: str | Path,
    *,
    output_path: str | Path | None = None,
    expected_metrics: list[str] | None = None,
    task_id: str = "sample",
) -> Path:
    run_path = resolve_run_dir(run_dir)
    result_path = run_path / "task_results.jsonl"
    if not result_path.exists():
        raise FileNotFoundError(f"Missing task_results.jsonl: {result_path}")

    expected = list(expected_metrics or infer_expected_metrics(run_path))
    if not expected:
        raise ValueError("Expected metric list must not be empty.")

    rows = read_jsonl(result_path)
    grouped = group_result_rows(rows, expected_metrics=expected, task_id=task_id)
    out_path = Path(output_path) if output_path is not None else run_path / "sample_metric_results.jsonl"
    out_path = out_path if out_path.is_absolute() else (Path.cwd() / out_path).resolve()
    write_jsonl(out_path, grouped, append=False)
    return out_path


def group_result_rows(
    rows: Iterable[dict[str, Any]],
    *,
    expected_metrics: list[str],
    task_id: str = "sample",
) -> list[dict[str, Any]]:
    by_sample: OrderedDict[str, dict[str, dict[str, Any]]] = OrderedDict()
    expected_set = set(expected_metrics)

    for row in rows:
        sample_id = row.get("sample_id")
        metric = row.get("metric")
        row_task_id = row.get("task_id")
        if sample_id is None or metric is None:
            continue
        if str(row_task_id) != task_id:
            continue
        metric_name = str(metric)
        if metric_name not in expected_set:
            continue
        sample_key = str(sample_id)
        by_sample.setdefault(sample_key, {})[metric_name] = row

    return [
        build_sample_row(sample_id, metric_rows, expected_metrics)
        for sample_id, metric_rows in by_sample.items()
    ]


def build_sample_row(
    sample_id: str,
    metric_rows: dict[str, dict[str, Any]],
    expected_metrics: list[str],
) -> dict[str, Any]:
    metrics: OrderedDict[str, dict[str, Any]] = OrderedDict()
    scores: list[float] = []
    missing_metrics: list[str] = []
    invalid_metrics: list[str] = []

    for metric in expected_metrics:
        row = metric_rows.get(metric)
        if row is None:
            missing_metrics.append(metric)
            continue

        metrics[metric] = strip_sample_id(row)
        score = normalize_score(row.get("score"))
        if score is None or row.get("error") is not None:
            invalid_metrics.append(metric)
        else:
            scores.append(score)

    data_quality: dict[str, Any] = {
        "metric": "data_quality",
        "score": min(scores) if not missing_metrics and not invalid_metrics else None,
        "aggregation": "min",
        "source_metrics": expected_metrics,
        "missing_metrics": missing_metrics,
        "invalid_metrics": invalid_metrics,
    }
    return {
        "sample_id": sample_id,
        "metrics": metrics,
        "data_quality": data_quality,
    }


def strip_sample_id(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key != "sample_id"}


def normalize_score(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return float(value) if value in {0, 1, 2} else None
    if isinstance(value, float):
        numeric = int(value)
        return float(numeric) if value == numeric and numeric in {0, 1, 2} else None
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        try:
            numeric_float = float(raw)
        except ValueError:
            return None
        numeric = int(numeric_float)
        return float(numeric) if numeric_float == numeric and numeric in {0, 1, 2} else None
    return None


def infer_expected_metrics(run_dir: Path) -> list[str]:
    config_path = find_run_config(run_dir)
    if config_path is None:
        return list(DEFAULT_EXPECTED_METRICS)

    config = load_config(config_path)
    metrics = []
    for entry in config.get("metrics") or []:
        if isinstance(entry, str):
            metrics.append(entry)
        elif isinstance(entry, dict) and entry.get("name"):
            metrics.append(str(entry["name"]))
    return metrics or list(DEFAULT_EXPECTED_METRICS)


def find_run_config(run_dir: Path) -> Path | None:
    for name in ("run_config.yaml", "run_config.yml", "run_config.json"):
        path = run_dir / name
        if path.exists():
            return path
    matches = sorted(run_dir.glob("run_config.*"))
    return matches[0] if matches else None


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


if __name__ == "__main__":
    main()

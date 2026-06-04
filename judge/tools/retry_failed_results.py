#!/usr/bin/env python3
"""Retry failed judge results with a retry config and write a merged run."""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
import time
from collections import Counter, defaultdict
from glob import glob
from pathlib import Path
from typing import Any, Iterable, Iterator


SCRIPT_PATH = Path(__file__).resolve()
TOOLS_ROOT = SCRIPT_PATH.parent
JUDGE_ROOT = TOOLS_ROOT.parent
OUTPUTS_ROOT = JUDGE_ROOT / "outputs"
REPO_ROOT = JUDGE_ROOT.parent
SUPPORTED_SUFFIXES = (".jsonl.gz", ".jsonl", ".json", ".gz")

sys.path.insert(0, str(JUDGE_ROOT))

from core.data import iter_records, to_judge_case  # noqa: E402
from core.io import append_jsonl, load_config, read_jsonl  # noqa: E402
from core.registry import load_configured_metrics  # noqa: E402
from core.runner import JudgeRunner, _build_client  # noqa: E402
from core.schemas import JudgeResult  # noqa: E402


ResultKey = tuple[str, str, str]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Retry score-null/error judge results with a higher-budget config, "
            "then write a judge-run-compatible merged output directory."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run-dir", required=True, help="Original judge run directory.")
    parser.add_argument(
        "--retry-config",
        required=True,
        help="YAML/JSON config containing retry overrides, usually client.max_tokens.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Merged retry run directory. Defaults to <run-dir>_retry.",
    )
    parser.add_argument(
        "--metric",
        action="append",
        default=None,
        help="Metric name to retry. May be passed more than once.",
    )
    parser.add_argument(
        "--error-types",
        nargs="+",
        default=None,
        help=(
            "Only retry rows whose error.type is in this list. Defaults to "
            "retry.error_types from retry config, or all score-null/error rows."
        ),
    )
    parser.add_argument(
        "--input",
        nargs="+",
        default=None,
        help="Override source input paths. Defaults to input.paths from run_config.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting files in an existing output directory.",
    )
    args = parser.parse_args(argv)

    run_dir = resolve_run_dir(args.run_dir)
    retry_config_path = resolve_path(args.retry_config)
    output_dir = (
        resolve_path(args.output_dir)
        if args.output_dir
        else run_dir.with_name(f"{run_dir.name}_retry")
    )
    prepare_output_dir(output_dir, overwrite=args.overwrite)

    original_config_path = find_run_config(run_dir)
    original_config = load_config(original_config_path)
    retry_config = load_config(retry_config_path)
    effective_config = deep_merge(original_config, retry_config)
    metric_filter = set(args.metric or [])
    retry_section = retry_config.get("retry") or {}
    error_types = set(
        args.error_types
        if args.error_types is not None
        else retry_section.get("error_types") or []
    )

    original_results = read_jsonl(run_dir / "task_results.jsonl")
    failed_keys = retryable_keys(
        original_results,
        metric_filter=metric_filter,
        error_types=error_types,
    )
    if not failed_keys:
        write_compatible_copy(
            run_dir=run_dir,
            output_dir=output_dir,
            original_config_path=original_config_path,
            retry_config_path=retry_config_path,
            manifest={
                "original_run_dir": str(run_dir),
                "output_dir": str(output_dir),
                "retry_config": str(retry_config_path),
                "retryable_count": 0,
                "attempted_count": 0,
                "replaced_count": 0,
                "still_failed_count": 0,
            },
        )
        print("[retry] retryable results: 0")
        print(f"[retry] output_dir: {output_dir}")
        return

    input_patterns = (
        args.input
        if args.input is not None
        else (original_config.get("input") or {}).get("paths") or []
    )
    source_paths = expand_input_paths(input_patterns, run_dir=run_dir)
    if not source_paths:
        raise FileNotFoundError(f"No source input files matched: {input_patterns}")

    cases, missing = load_cases_for_keys(failed_keys, source_paths)
    if missing:
        print(f"[retry] missing source samples: {len(missing)}", file=sys.stderr)
        for sample_id in sorted(missing):
            print(f"[retry] missing: {sample_id}", file=sys.stderr)

    copy_run_configs(original_config_path, retry_config_path, output_dir)
    retry_results = run_retry_tasks(
        cases=cases,
        failed_keys=failed_keys,
        config=effective_config,
        output_dir=output_dir,
    )
    move_retry_only_outputs(output_dir)

    retry_rows = [result.to_dict() for result in retry_results]
    retry_result_path = output_dir / "task_results.retry.jsonl"
    retry_rows = read_jsonl(retry_result_path) if retry_result_path.exists() else retry_rows
    merged_results, merge_manifest = merge_retry_results(
        original_results,
        retry_rows,
        retry_run_dir=output_dir,
        eligible_keys=failed_keys,
    )
    write_jsonl(output_dir / "task_results.jsonl", merged_results)

    original_raw = read_jsonl(run_dir / "raw_responses.jsonl")
    retry_raw = read_jsonl(output_dir / "raw_responses.retry.jsonl")
    write_merged_raw_responses(output_dir / "raw_responses.jsonl", merged_results, original_raw, retry_raw)
    write_summary(output_dir / "summary.json", merged_results, merge_manifest)
    manifest = {
        "created_at": int(time.time()),
        "original_run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "retry_config": str(retry_config_path),
        "source_files": [str(path) for path in source_paths],
        "retryable_count": len(failed_keys),
        "attempted_count": len(retry_rows),
        "missing_source_sample_ids": sorted(missing),
        **merge_manifest,
    }
    (output_dir / "retry_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"[retry] original_run_dir: {run_dir}")
    print(f"[retry] output_dir: {output_dir}")
    print(f"[retry] retryable results: {len(failed_keys)}")
    print(f"[retry] retry attempts: {len(retry_rows)}")
    print(f"[retry] replaced results: {merge_manifest['replaced_count']}")
    print(f"[retry] still failed results: {merge_manifest['still_failed_count']}")


def resolve_run_dir(raw: str) -> Path:
    path = Path(raw)
    candidates = [path, OUTPUTS_ROOT / path, JUDGE_ROOT / path, REPO_ROOT / path]
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate.resolve()
    raise FileNotFoundError(f"Run directory not found: {raw}")


def resolve_path(raw: str | None) -> Path:
    if raw is None:
        raise ValueError("Path must not be None")
    path = Path(raw)
    return path if path.is_absolute() else (Path.cwd() / path).resolve()


def prepare_output_dir(output_dir: Path, *, overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    managed_names = {
        "task_results.jsonl",
        "task_results.retry.jsonl",
        "raw_responses.jsonl",
        "raw_responses.retry.jsonl",
        "summary.json",
        "run_config.yaml",
        "retry_config.yaml",
        "retry_manifest.json",
    }
    existing = [name for name in managed_names if (output_dir / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"Output directory already contains retry output files: {output_dir}. "
            "Use --overwrite to replace them."
        )
    if overwrite:
        for name in existing:
            (output_dir / name).unlink()


def find_run_config(run_dir: Path) -> Path:
    for name in ("run_config.yaml", "run_config.yml", "run_config.json"):
        path = run_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(f"No run_config.yaml/json found in {run_dir}")


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if (
            isinstance(value, dict)
            and isinstance(merged.get(key), dict)
        ):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def retryable_keys(
    rows: list[dict[str, Any]],
    *,
    metric_filter: set[str],
    error_types: set[str],
) -> set[ResultKey]:
    keys: set[ResultKey] = set()
    for row in rows:
        sample_id = str(row.get("sample_id") or "")
        metric = str(row.get("metric") or "")
        task_id = str(row.get("task_id") or "")
        if not sample_id or not metric or not task_id:
            continue
        if metric_filter and metric not in metric_filter:
            continue
        error = row.get("error")
        failed = row.get("score") is None or error is not None
        if not failed:
            continue
        if error_types:
            error_type = error.get("type") if isinstance(error, dict) else None
            if error_type not in error_types:
                continue
        keys.add((sample_id, metric, task_id))
    return keys


def expand_input_paths(patterns: Iterable[str], *, run_dir: Path) -> list[Path]:
    roots = [Path.cwd(), REPO_ROOT, JUDGE_ROOT, OUTPUTS_ROOT, run_dir]
    paths: list[Path] = []
    for pattern in patterns:
        raw_path = Path(pattern)
        candidates = [raw_path] if raw_path.is_absolute() else [root / raw_path for root in roots]
        for candidate in candidates:
            matches = glob(str(candidate), recursive=True)
            if matches:
                for match in matches:
                    path = Path(match)
                    if path.is_dir():
                        paths.extend(iter_supported_files(path))
                    elif is_supported(path):
                        paths.append(path)
                continue
            if candidate.is_dir():
                paths.extend(iter_supported_files(candidate))
            elif candidate.exists() and is_supported(candidate):
                paths.append(candidate)
    return sorted(set(path.resolve() for path in paths))


def iter_supported_files(root: Path) -> Iterator[Path]:
    for pattern in ("*.jsonl.gz", "*.jsonl", "*.json"):
        yield from root.rglob(pattern)


def is_supported(path: Path) -> bool:
    name = path.name.lower()
    return any(name.endswith(suffix) for suffix in SUPPORTED_SUFFIXES)


def load_cases_for_keys(
    failed_keys: set[ResultKey],
    source_paths: list[Path],
) -> tuple[list[Any], set[str]]:
    wanted_ids = {key[0] for key in failed_keys}
    found: dict[str, Any] = {}
    source_index = 0
    for path in source_paths:
        for source_row, record in enumerate(iter_records(path)):
            try:
                case = to_judge_case(
                    record,
                    source_index=source_index,
                    source_path=str(path),
                    source_row=source_row,
                )
            except Exception:
                source_index += 1
                continue
            if case.sample_id in wanted_ids and case.sample_id not in found:
                found[case.sample_id] = case
                if len(found) == len(wanted_ids):
                    break
            source_index += 1
        if len(found) == len(wanted_ids):
            break
    missing = wanted_ids - set(found)
    return list(found.values()), missing


def copy_run_configs(
    original_config_path: Path,
    retry_config_path: Path,
    output_dir: Path,
) -> None:
    shutil.copyfile(original_config_path, output_dir / "run_config.yaml")
    shutil.copyfile(retry_config_path, output_dir / "retry_config.yaml")


def run_retry_tasks(
    *,
    cases: list[Any],
    failed_keys: set[ResultKey],
    config: dict[str, Any],
    output_dir: Path,
) -> list[JudgeResult]:
    judge_root = JUDGE_ROOT
    metric_entries = config.get("metrics") or []
    metrics_root = Path(config.get("metrics_root") or judge_root / "metrics")
    if not metrics_root.is_absolute():
        metrics_root = judge_root / metrics_root
    metrics = load_configured_metrics(metrics_root, metric_entries)
    client_config = config.get("client") or {}
    client = _build_client(client_config)
    runner = JudgeRunner(
        metrics=metrics,
        client=client,
        output_dir=output_dir,
        client_config=client_config,
        resume=False,
    )

    all_results: list[JudgeResult] = []
    for metric in metrics:
        tasks = []
        for case in cases:
            for task in metric.build_tasks(case):
                if (task.sample_id, task.metric, task.task_id) in failed_keys:
                    tasks.append(task)
        if not tasks:
            continue
        print(f"[retry] {metric.config.name}: {len(tasks)} retry tasks")
        all_results.extend(runner._run_metric(metric, tasks))
    return all_results


def move_retry_only_outputs(output_dir: Path) -> None:
    result_path = output_dir / "task_results.jsonl"
    raw_path = output_dir / "raw_responses.jsonl"
    retry_result_path = output_dir / "task_results.retry.jsonl"
    retry_raw_path = output_dir / "raw_responses.retry.jsonl"
    if result_path.exists():
        result_path.replace(retry_result_path)
    else:
        retry_result_path.write_text("", encoding="utf-8")
    if raw_path.exists():
        raw_path.replace(retry_raw_path)
    else:
        retry_raw_path.write_text("", encoding="utf-8")


def merge_retry_results(
    original_rows: list[dict[str, Any]],
    retry_rows: list[dict[str, Any]],
    *,
    retry_run_dir: Path,
    eligible_keys: set[ResultKey] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    successful_retry_by_key = {
        result_key(row): row
        for row in retry_rows
        if result_key(row) is not None and row.get("score") is not None and row.get("error") is None
    }
    retry_failed_keys = {
        key
        for row in retry_rows
        if (key := result_key(row)) is not None
        and (row.get("score") is None or row.get("error") is not None)
    }
    original_failed_keys = {
        key
        for row in original_rows
        if (key := result_key(row)) is not None
        and (row.get("score") is None or row.get("error") is not None)
        and (eligible_keys is None or key in eligible_keys)
    }

    merged: list[dict[str, Any]] = []
    replaced_keys: list[ResultKey] = []
    for row in original_rows:
        key = result_key(row)
        replacement = successful_retry_by_key.get(key)
        if key is not None and key in original_failed_keys and replacement is not None:
            merged_row = copy.deepcopy(replacement)
            merged_row["retry_from"] = {
                "original_raw_response_id": row.get("raw_response_id"),
                "original_params_hash": row.get("params_hash"),
                "original_error": row.get("error"),
                "retry_run_dir": str(retry_run_dir),
            }
            merged.append(merged_row)
            replaced_keys.append(key)
        else:
            merged.append(copy.deepcopy(row))

    replaced = set(replaced_keys)
    still_failed = (original_failed_keys - replaced) | retry_failed_keys
    return merged, {
        "replaced_count": len(replaced),
        "still_failed_count": len(still_failed),
        "replaced_keys": [key_to_dict(key) for key in sorted(replaced)],
        "still_failed_keys": [key_to_dict(key) for key in sorted(still_failed)],
    }


def result_key(row: dict[str, Any]) -> ResultKey | None:
    sample_id = row.get("sample_id")
    metric = row.get("metric")
    task_id = row.get("task_id")
    if sample_id is None or metric is None or task_id is None:
        return None
    return str(sample_id), str(metric), str(task_id)


def key_to_dict(key: ResultKey) -> dict[str, str]:
    sample_id, metric, task_id = key
    return {
        "sample_id": sample_id,
        "metric": metric,
        "task_id": task_id,
    }


def write_merged_raw_responses(
    path: Path,
    merged_results: list[dict[str, Any]],
    original_raw: list[dict[str, Any]],
    retry_raw: list[dict[str, Any]],
) -> None:
    raw_by_id = {
        row.get("raw_response_id"): row
        for row in [*original_raw, *retry_raw]
        if row.get("raw_response_id")
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for result in merged_results:
            raw_id = result.get("raw_response_id")
            raw_row = raw_by_id.get(raw_id)
            if raw_row is None:
                continue
            f.write(json.dumps(raw_row, ensure_ascii=False) + "\n")


def write_summary(
    path: Path,
    merged_results: list[dict[str, Any]],
    merge_manifest: dict[str, Any],
) -> None:
    by_metric: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in merged_results:
        by_metric[str(row.get("metric") or "unknown")].append(row)
    metric_summary = {}
    for metric, rows in by_metric.items():
        scores = [row.get("score") for row in rows if row.get("score") is not None]
        metric_summary[metric] = {
            "n_results": len(rows),
            "n_scored": len(scores),
            "score_mean": sum(scores) / len(scores) if scores else None,
            "n_errors": sum(1 for row in rows if row.get("error") is not None),
            "score_counts": dict(Counter(str(row.get("score")) for row in rows)),
        }
    summary = {
        "created_at": int(time.time()),
        "total_tasks": len(merged_results),
        "skipped_tasks": 0,
        "written_results": len(merged_results),
        "metrics": metric_summary,
        "retry": {
            "replaced_count": merge_manifest.get("replaced_count", 0),
            "still_failed_count": merge_manifest.get("still_failed_count", 0),
        },
    }
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def write_compatible_copy(
    *,
    run_dir: Path,
    output_dir: Path,
    original_config_path: Path,
    retry_config_path: Path,
    manifest: dict[str, Any],
) -> None:
    copy_run_configs(original_config_path, retry_config_path, output_dir)
    for name in ("task_results.jsonl", "raw_responses.jsonl", "summary.json"):
        source = run_dir / name
        if source.exists():
            shutil.copyfile(source, output_dir / name)
    (output_dir / "task_results.retry.jsonl").write_text("", encoding="utf-8")
    (output_dir / "raw_responses.retry.jsonl").write_text("", encoding="utf-8")
    (output_dir / "retry_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

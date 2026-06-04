#!/usr/bin/env python3
"""Extract source samples where two judge runs assign different scores."""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from glob import glob
from pathlib import Path
from typing import Any, Iterable, Iterator


SCRIPT_PATH = Path(__file__).resolve()
TOOLS_ROOT = SCRIPT_PATH.parent
JUDGE_ROOT = TOOLS_ROOT.parent
OUTPUTS_ROOT = JUDGE_ROOT / "outputs"
REPO_ROOT = JUDGE_ROOT.parent
SUPPORTED_SUFFIXES = (".jsonl.gz", ".jsonl", ".json", ".gz")
RESULT_FIELDS = (
    "metric",
    "metric_version",
    "task_id",
    "score",
    "passed",
    "reason",
    "details",
    "error",
    "raw_response_id",
    "params_hash",
)

sys.path.insert(0, str(JUDGE_ROOT))

from core.data import iter_records, to_judge_case  # noqa: E402
from core.io import load_config, read_jsonl  # noqa: E402


ResultKey = tuple[str, str, str]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare two judge runs, find score disagreements by "
            "sample_id/metric/task_id, and export original samples with both "
            "models' judge results attached."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run-a", required=True, help="First judge run directory.")
    parser.add_argument("--run-b", required=True, help="Second judge run directory.")
    parser.add_argument("--model-a", default=None, help="Model name for --run-a.")
    parser.add_argument("--model-b", default=None, help="Model name for --run-b.")
    parser.add_argument(
        "--metric",
        action="append",
        default=None,
        help=(
            "Metric name to compare. May be passed more than once. "
            "Defaults to all common metrics."
        ),
    )
    parser.add_argument(
        "--input",
        nargs="+",
        default=None,
        help=(
            "Override source input paths. Defaults to input.paths from both "
            "run_config files."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path. Defaults to <run-a>/score_disagreements_vs_<model-b>.<format>.",
    )
    parser.add_argument(
        "--format",
        choices=["jsonl", "json"],
        default="jsonl",
        help="Output file format.",
    )
    parser.add_argument(
        "--include-raw",
        action="store_true",
        help="Include the full original canonical record under the raw field.",
    )
    args = parser.parse_args(argv)

    run_a = resolve_run_dir(args.run_a)
    run_b = resolve_run_dir(args.run_b)
    config_a = load_config(find_run_config(run_a))
    config_b = load_config(find_run_config(run_b))
    model_a = args.model_a or infer_model_name(run_a)
    model_b = args.model_b or infer_model_name(run_b)
    metric_filter = set(args.metric or [])

    results_a = load_result_index(run_a, metric_filter=metric_filter)
    results_b = load_result_index(run_b, metric_filter=metric_filter)
    disagreements = find_score_disagreements(results_a, results_b)
    if not disagreements:
        out_path = output_path(args.output, run_a, model_b, args.format)
        write_rows(out_path, [], fmt=args.format)
        print(f"[compare] run_a: {run_a}")
        print(f"[compare] run_b: {run_b}")
        print("[compare] disagreements: 0")
        print(f"[compare] output: {out_path}")
        return

    input_patterns = (
        args.input
        if args.input is not None
        else combined_input_paths(config_a, config_b)
    )
    source_paths = expand_input_paths(input_patterns, run_dirs=[run_a, run_b])
    if not source_paths:
        raise FileNotFoundError(f"No source input files matched: {input_patterns}")

    rows, missing = build_disagreement_rows(
        disagreements,
        results_a,
        results_b,
        source_paths,
        model_a=model_a,
        model_b=model_b,
        include_raw=args.include_raw,
    )
    out_path = output_path(args.output, run_a, model_b, args.format)
    write_rows(out_path, rows, fmt=args.format)

    print(f"[compare] run_a: {run_a}")
    print(f"[compare] run_b: {run_b}")
    print(f"[compare] model_a: {model_a}")
    print(f"[compare] model_b: {model_b}")
    print(f"[compare] source files: {len(source_paths)}")
    print(f"[compare] disagreements: {len(disagreements)}")
    print(f"[compare] samples written: {len(rows)}")
    print(f"[compare] output: {out_path}")
    if missing:
        print(f"[compare] missing source samples: {len(missing)}", file=sys.stderr)
        for sample_id in missing:
            print(f"[compare] missing: {sample_id}", file=sys.stderr)


def resolve_run_dir(raw: str) -> Path:
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


def find_run_config(run_dir: Path) -> Path:
    for name in ("run_config.yaml", "run_config.yml", "run_config.json"):
        path = run_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(f"No run_config.yaml/json found in {run_dir}")


def infer_model_name(run_dir: Path) -> str:
    parent = run_dir.parent.name.strip()
    return safe_model_name(parent or run_dir.name)


def safe_model_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value) or "model"


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
        sample_id = str(row.get("sample_id") or "")
        metric = str(row.get("metric") or "")
        task_id = str(row.get("task_id") or "")
        if not sample_id or not metric or not task_id:
            continue
        if metric_filter and metric not in metric_filter:
            continue
        indexed[(sample_id, metric, task_id)] = row
    return indexed


def find_score_disagreements(
    results_a: dict[ResultKey, dict[str, Any]],
    results_b: dict[ResultKey, dict[str, Any]],
) -> list[ResultKey]:
    common_keys = set(results_a) & set(results_b)
    disagreements = [
        key
        for key in common_keys
        if results_a[key].get("score") != results_b[key].get("score")
    ]
    return sorted(disagreements, key=lambda key: (key[1], key[2], key[0]))


def combined_input_paths(config_a: dict[str, Any], config_b: dict[str, Any]) -> list[str]:
    paths: list[str] = []
    seen: set[str] = set()
    for config in (config_a, config_b):
        for path in (config.get("input") or {}).get("paths") or []:
            raw = str(path)
            if raw not in seen:
                seen.add(raw)
                paths.append(raw)
    return paths


def expand_input_paths(patterns: Iterable[str], *, run_dirs: list[Path]) -> list[Path]:
    roots = [Path.cwd(), REPO_ROOT, JUDGE_ROOT, OUTPUTS_ROOT, *run_dirs]
    paths: list[Path] = []
    for pattern in patterns:
        raw_path = Path(pattern)
        candidates = [raw_path] if raw_path.is_absolute() else [r / raw_path for r in roots]
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
    return sorted(set(p.resolve() for p in paths))


def iter_supported_files(root: Path) -> Iterator[Path]:
    for pattern in ("*.jsonl.gz", "*.jsonl", "*.json"):
        yield from root.rglob(pattern)


def is_supported(path: Path) -> bool:
    name = path.name.lower()
    return any(name.endswith(suffix) for suffix in SUPPORTED_SUFFIXES)


def build_disagreement_rows(
    disagreement_keys: list[ResultKey],
    results_a: dict[ResultKey, dict[str, Any]],
    results_b: dict[ResultKey, dict[str, Any]],
    source_paths: list[Path],
    *,
    model_a: str,
    model_b: str,
    include_raw: bool,
) -> tuple[list[dict[str, Any]], list[str]]:
    by_sample: dict[str, list[ResultKey]] = {}
    for key in disagreement_keys:
        by_sample.setdefault(key[0], []).append(key)

    found: set[str] = set()
    rows: list[dict[str, Any]] = []
    for path in source_paths:
        for record in safe_iter_records(path):
            try:
                case = to_judge_case(record)
            except Exception:
                continue
            if case.sample_id not in by_sample or case.sample_id in found:
                continue
            rows.append(source_row(
                case=case,
                source_path=path,
                keys=by_sample[case.sample_id],
                results_a=results_a,
                results_b=results_b,
                model_a=model_a,
                model_b=model_b,
                include_raw=include_raw,
            ))
            found.add(case.sample_id)
            if len(found) == len(by_sample):
                break
        if len(found) == len(by_sample):
            break

    missing = [sample_id for sample_id in by_sample if sample_id not in found]
    return rows, missing


def source_row(
    *,
    case: Any,
    source_path: Path,
    keys: list[ResultKey],
    results_a: dict[ResultKey, dict[str, Any]],
    results_b: dict[ResultKey, dict[str, Any]],
    model_a: str,
    model_b: str,
    include_raw: bool,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "sample_id": case.sample_id,
        "query": case.query,
        "conversations": case.conversations,
        "tools": case.tools,
        "metadata": case.metadata or {},
        "source_path": str(source_path),
        "judge_disagreements": [],
        "judge_results": [],
    }
    if include_raw:
        row["raw"] = case.raw

    for key in sorted(keys, key=lambda item: (item[1], item[2])):
        _, metric, task_id = key
        result_a = trim_result(results_a[key], model=model_a)
        result_b = trim_result(results_b[key], model=model_b)
        row["judge_disagreements"].append({
            "metric": metric,
            "task_id": task_id,
            "scores": {
                model_a: result_a.get("score"),
                model_b: result_b.get("score"),
            },
        })
        row["judge_results"].extend([result_a, result_b])

    if len(row["judge_disagreements"]) == 1:
        row["judge_disagreement"] = row["judge_disagreements"][0]
    return row


def trim_result(result: dict[str, Any], *, model: str) -> dict[str, Any]:
    out = {"model": model}
    for field in RESULT_FIELDS:
        out[field] = result.get(field)
    return out


def safe_iter_records(path: Path) -> Iterator[dict[str, Any]]:
    try:
        yield from iter_records(path)
    except gzip.BadGzipFile as exc:
        raise RuntimeError(f"Bad gzip file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Bad JSON in source file: {path}") from exc


def output_path(raw_output: str | None, run_a: Path, model_b: str, fmt: str) -> Path:
    if raw_output:
        return Path(raw_output)
    return run_a / f"score_disagreements_vs_{safe_model_name(model_b)}.{fmt}"


def write_rows(path: Path, rows: list[dict[str, Any]], *, fmt: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "json":
        path.write_text(
            json.dumps(rows, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

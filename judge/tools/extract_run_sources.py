#!/usr/bin/env python3
"""Extract original canonical samples used by a judge output run."""

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

sys.path.insert(0, str(JUDGE_ROOT))

from core.data import iter_records, to_judge_case  # noqa: E402
from core.io import load_config, read_jsonl  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract query/conversations/tools for all sample ids referenced by "
            "a judge output directory."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "run_dir",
        help=(
            "Judge run directory. Accepts an absolute path, a path relative to "
            "the current directory, or a path relative to data_quality/judge/outputs "
            "(for example all_metrics/20260527_171039_785700)."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path. Defaults to <run_dir>/source_samples.<format>.",
    )
    parser.add_argument(
        "--input",
        nargs="+",
        default=None,
        help=(
            "Override input.paths from run_config. Accepts files, directories, "
            "or glob patterns."
        ),
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
    args = parser.parse_args()

    run_dir = resolve_run_dir(args.run_dir)
    config_path = find_run_config(run_dir)
    config = load_config(config_path)
    sample_ids = load_run_sample_ids(run_dir)
    if not sample_ids:
        raise RuntimeError(f"No sample ids found in {run_dir}")

    input_paths = args.input if args.input is not None else (config.get("input") or {}).get("paths") or []
    source_paths = expand_input_paths(input_paths, run_dir=run_dir)
    if not source_paths:
        raise FileNotFoundError(f"No source input files matched: {input_paths}")

    rows, missing = extract_samples(sample_ids, source_paths, include_raw=args.include_raw)
    out_path = (
        Path(args.output)
        if args.output
        else run_dir / f"source_samples.{args.format}"
    )
    write_rows(out_path, rows, fmt=args.format)

    print(f"[extract] run_dir: {run_dir}")
    print(f"[extract] run_config: {config_path}")
    print(f"[extract] source files: {len(source_paths)}")
    print(f"[extract] sample ids requested: {len(sample_ids)}")
    print(f"[extract] samples written: {len(rows)}")
    print(f"[extract] output: {out_path}")
    if missing:
        print(f"[extract] missing sample ids: {len(missing)}", file=sys.stderr)
        for sample_id in missing:
            print(f"[extract] missing: {sample_id}", file=sys.stderr)


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


def load_run_sample_ids(run_dir: Path) -> list[str]:
    rows = read_jsonl(run_dir / "task_results.jsonl")
    if not rows:
        rows = read_jsonl(run_dir / "raw_responses.jsonl")

    sample_ids: list[str] = []
    seen: set[str] = set()
    for row in rows:
        sample_id = str(row.get("sample_id") or "")
        if sample_id and sample_id not in seen:
            seen.add(sample_id)
            sample_ids.append(sample_id)
    return sample_ids


def expand_input_paths(patterns: Iterable[str], *, run_dir: Path) -> list[Path]:
    roots = [Path.cwd(), REPO_ROOT, JUDGE_ROOT, run_dir]
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


def extract_samples(
    sample_ids: list[str],
    source_paths: list[Path],
    *,
    include_raw: bool,
) -> tuple[list[dict[str, Any]], list[str]]:
    wanted = set(sample_ids)
    found: dict[str, dict[str, Any]] = {}

    for path in source_paths:
        for record in safe_iter_records(path):
            try:
                case = to_judge_case(record)
            except Exception:
                continue
            if case.sample_id not in wanted or case.sample_id in found:
                continue
            row: dict[str, Any] = {
                "sample_id": case.sample_id,
                "query": case.query,
                "conversations": case.conversations,
                "tools": case.tools,
                "metadata": case.metadata or {},
                "source_path": str(path),
            }
            if include_raw:
                row["raw"] = case.raw
            found[case.sample_id] = row
            if len(found) == len(wanted):
                break
        if len(found) == len(wanted):
            break

    rows = [found[sample_id] for sample_id in sample_ids if sample_id in found]
    missing = [sample_id for sample_id in sample_ids if sample_id not in found]
    return rows, missing


def safe_iter_records(path: Path) -> Iterator[dict[str, Any]]:
    try:
        yield from iter_records(path)
    except gzip.BadGzipFile as exc:
        raise RuntimeError(f"Bad gzip file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Bad JSON in source file: {path}") from exc


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

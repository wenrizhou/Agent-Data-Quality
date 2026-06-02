"""Canonical JSON/JSONL/JSONL.GZ loading helpers."""

from __future__ import annotations

import gzip
import json
import random
from glob import glob
from pathlib import Path
from typing import Any, Iterable, Iterator

from .schemas import JudgeCase


def expand_input_paths(patterns: Iterable[str]) -> list[Path]:
    """Expand files, directories, and glob patterns into sorted existing paths."""
    paths: list[Path] = []
    for pattern in patterns:
        matches = glob(pattern, recursive=True)
        if matches:
            for match in matches:
                p = Path(match)
                if p.is_dir():
                    paths.extend(_iter_supported_files(p))
                elif _is_supported_file(p):
                    paths.append(p)
        else:
            p = Path(pattern)
            if p.is_dir():
                paths.extend(_iter_supported_files(p))
            elif p.exists() and _is_supported_file(p):
                paths.append(p)
    return sorted(set(paths))


def _iter_supported_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for pattern in ("*.jsonl.gz", "*.jsonl", "*.json"):
        files.extend(root.rglob(pattern))
    return files


def _is_supported_file(path: Path) -> bool:
    name = path.name.lower()
    return (
        name.endswith(".jsonl.gz")
        or name.endswith(".jsonl")
        or name.endswith(".json")
        or name.endswith(".gz")
    )


def iter_records(path: Path) -> Iterator[dict[str, Any]]:
    """Yield dict records from .json, .jsonl, .jsonl.gz, or .gz files."""
    name = path.name.lower()
    if name.endswith(".jsonl.gz") or name.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                raw = line.strip()
                if raw:
                    yield json.loads(raw)
        return

    if name.endswith(".jsonl"):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                raw = line.strip()
                if raw:
                    yield json.loads(raw)
        return

    if name.endswith(".json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            yield from data
        elif isinstance(data, dict):
            yield data
        else:
            raise ValueError(f"Unsupported JSON top-level type in {path}")
        return

    raise ValueError(f"Unsupported input file type: {path}")


def to_judge_case(
    sample: dict[str, Any],
    *,
    source_index: int | None = None,
    source_path: str | None = None,
    source_row: int | None = None,
) -> JudgeCase:
    """Convert a canonical sample dict into JudgeCase."""
    sample_id = str(sample.get("id") or sample.get("sample_id") or "")
    if not sample_id:
        if source_index is None:
            raise ValueError("Sample missing id/sample_id")
        sample_id = f"auto_{source_index:06d}"

    conversations = sample.get("conversations", sample.get("messages", []))
    if not isinstance(conversations, list):
        conversations = []

    tools = sample.get("tools") or []
    if not isinstance(tools, list):
        tools = []

    metadata = sample.get("metadata")
    if metadata is not None and not isinstance(metadata, dict):
        metadata = {"value": metadata}
    metadata = dict(metadata or {})
    metadata.setdefault("_judge_source", {
        "source_index": source_index,
        "source_path": source_path,
        "source_row": source_row,
    })

    return JudgeCase(
        sample_id=sample_id,
        query=sample.get("query"),
        conversations=conversations,
        tools=tools,
        metadata=metadata,
        source_index=source_index,
        source_path=source_path,
        source_row=source_row,
        raw=sample,
    )


def load_cases(
    patterns: Iterable[str],
    max_samples: int | None = None,
    *,
    balanced_sample: bool = False,
    seed: int = 42,
) -> list[JudgeCase]:
    """Load canonical records from path patterns."""
    paths = expand_input_paths(patterns)
    if not paths:
        raise FileNotFoundError(f"No input files matched: {list(patterns)}")

    if balanced_sample and max_samples is not None:
        return _load_cases_balanced(paths, max_samples=max_samples, seed=seed)

    cases: list[JudgeCase] = []
    for path in paths:
        for source_row, sample in enumerate(iter_records(path)):
            cases.append(
                to_judge_case(
                    sample,
                    source_index=len(cases),
                    source_path=str(path),
                    source_row=source_row,
                )
            )
            if max_samples is not None and len(cases) >= max_samples:
                return cases
    return cases


def _load_cases_balanced(
    paths: list[Path],
    *,
    max_samples: int,
    seed: int,
) -> list[JudgeCase]:
    """Sample approximately equal numbers of records from each input file."""
    if max_samples <= 0:
        return []

    counts = [_count_records(path) for path in paths]
    sample_counts = _allocate_balanced_counts(counts, max_samples, seed=seed)
    rng = random.Random(seed)
    selected_rows: dict[Path, set[int]] = {}
    for path, count, sample_count in zip(paths, counts, sample_counts):
        if sample_count <= 0:
            selected_rows[path] = set()
        elif sample_count >= count:
            selected_rows[path] = set(range(count))
        else:
            selected_rows[path] = set(rng.sample(range(count), sample_count))

    cases: list[JudgeCase] = []
    for path in paths:
        rows = selected_rows.get(path, set())
        if not rows:
            continue
        for source_row, sample in enumerate(iter_records(path)):
            if source_row not in rows:
                continue
            cases.append(
                to_judge_case(
                    sample,
                    source_index=len(cases),
                    source_path=str(path),
                    source_row=source_row,
                )
            )
    return cases


def _count_records(path: Path) -> int:
    return sum(1 for _ in iter_records(path))


def _allocate_balanced_counts(
    counts: list[int],
    max_samples: int,
    *,
    seed: int,
) -> list[int]:
    if not counts or max_samples <= 0:
        return [0 for _ in counts]

    target_total = min(max_samples, sum(counts))
    n_files = len(counts)
    base = target_total // n_files
    allocated = [min(count, base) for count in counts]
    remaining = target_total - sum(allocated)

    rng = random.Random(seed)
    candidates = [i for i, count in enumerate(counts) if allocated[i] < count]
    while remaining > 0 and candidates:
        rng.shuffle(candidates)
        progressed = False
        for idx in candidates:
            if remaining <= 0:
                break
            if allocated[idx] >= counts[idx]:
                continue
            allocated[idx] += 1
            remaining -= 1
            progressed = True
        candidates = [i for i in candidates if allocated[i] < counts[i]]
        if not progressed:
            break
    return allocated

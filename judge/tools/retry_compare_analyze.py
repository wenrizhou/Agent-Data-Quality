#!/usr/bin/env python3
"""Run retry, score-disagreement extraction, and disagreement analysis."""

from __future__ import annotations

import argparse
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


SCRIPT_PATH = Path(__file__).resolve()
TOOLS_ROOT = SCRIPT_PATH.parent
JUDGE_ROOT = TOOLS_ROOT.parent
REPO_ROOT = JUDGE_ROOT.parent

sys.path.insert(0, str(TOOLS_ROOT))

import extract_score_disagreements  # noqa: E402
import retry_failed_results  # noqa: E402
import summarize_score_disagreements  # noqa: E402


RETRY_MANAGED_NAMES = {
    "task_results.jsonl",
    "task_results.retry.jsonl",
    "raw_responses.jsonl",
    "raw_responses.retry.jsonl",
    "summary.json",
    "run_config.yaml",
    "retry_config.yaml",
    "retry_manifest.json",
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Retry failed results from two judge runs, compare score disagreements, "
            "then write a grouped disagreement report with optional LLM analysis."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run-a", required=True, help="First original judge run directory.")
    parser.add_argument("--run-b", required=True, help="Second original judge run directory.")
    parser.add_argument("--retry-config", required=True, help="Retry YAML/JSON config.")
    parser.add_argument(
        "--metric",
        action="append",
        default=None,
        help="Metric to retry/compare. May be passed more than once. Defaults to all metrics.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Disagreement JSONL/JSON path. Defaults to the common output root.",
    )
    parser.add_argument(
        "--analysis-output",
        default=None,
        help="Markdown report path. Defaults to <output-stem>_analysis.md.",
    )
    parser.add_argument(
        "--analysis-jsonl-output",
        default=None,
        help="Structured JSONL report path. Defaults to <output-stem>_analysis.jsonl.",
    )
    parser.add_argument(
        "--retry-output-a",
        default=None,
        help="Retry output directory for --run-a. Defaults to <run-a>_retry.",
    )
    parser.add_argument(
        "--retry-output-b",
        default=None,
        help="Retry output directory for --run-b. Defaults to <run-b>_retry.",
    )
    parser.add_argument("--model-a", default=None, help="MODEL override for retrying --run-a.")
    parser.add_argument("--model-b", default=None, help="MODEL override for retrying --run-b.")
    parser.add_argument("--base-url", default=None, help="Common BASE_URL override.")
    parser.add_argument("--api-key", default=None, help="Common API_KEY override.")
    parser.add_argument("--base-url-a", default=None, help="BASE_URL override for --run-a.")
    parser.add_argument("--api-key-a", default=None, help="API_KEY override for --run-a.")
    parser.add_argument("--base-url-b", default=None, help="BASE_URL override for --run-b.")
    parser.add_argument("--api-key-b", default=None, help="API_KEY override for --run-b.")
    parser.add_argument(
        "--analysis-model",
        default=None,
        help="MODEL override for LLM disagreement analysis. Defaults to env MODEL or model-b.",
    )
    parser.add_argument(
        "--analysis-base-url",
        default=None,
        help="BASE_URL override for LLM disagreement analysis.",
    )
    parser.add_argument(
        "--analysis-api-key",
        default=None,
        help="API_KEY override for LLM disagreement analysis.",
    )
    parser.add_argument(
        "--analysis-max-tokens",
        type=int,
        default=512,
        help="Max output tokens for each analysis request.",
    )
    parser.add_argument(
        "--analysis-timeout",
        type=float,
        default=120.0,
        help="Timeout for each analysis request.",
    )
    parser.add_argument(
        "--skip-analysis",
        action="store_true",
        help="Write grouped reports without calling an LLM for disagreement summaries.",
    )
    parser.add_argument(
        "--reuse-retry",
        action="store_true",
        help="Reuse an existing retry output directory when it has task_results.jsonl.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing retry/disagreement/report outputs.",
    )
    parser.add_argument(
        "--error-types",
        nargs="+",
        default=None,
        help="Only retry these error.type values. Defaults to retry config.",
    )
    parser.add_argument(
        "--input",
        nargs="+",
        default=None,
        help="Override source input paths for retry and disagreement extraction.",
    )
    parser.add_argument(
        "--format",
        choices=["jsonl", "json"],
        default="jsonl",
        help="Disagreement output format.",
    )
    parser.add_argument(
        "--include-raw",
        action="store_true",
        help="Include the full original canonical record under raw in disagreement output.",
    )
    original_group = parser.add_mutually_exclusive_group()
    original_group.add_argument(
        "--include-original",
        action="store_true",
        help="Include original data in the Markdown analysis report.",
    )
    original_group.add_argument(
        "--no-original",
        dest="include_original",
        action="store_false",
        help="Do not include original data in the Markdown analysis report.",
    )
    parser.set_defaults(include_original=True)
    args = parser.parse_args(argv)

    run_a = retry_failed_results.resolve_run_dir(args.run_a)
    run_b = retry_failed_results.resolve_run_dir(args.run_b)
    retry_config = resolve_path(args.retry_config)
    retry_a = resolve_output_dir(args.retry_output_a, run_a)
    retry_b = resolve_output_dir(args.retry_output_b, run_b)
    model_a = args.model_a or infer_model_from_run_dir(run_a)
    model_b = args.model_b or infer_model_from_run_dir(run_b)
    disagreement_output = resolve_disagreement_output(
        args.output,
        run_a=run_a,
        run_b=run_b,
        metrics=args.metric or [],
        fmt=args.format,
    )
    analysis_output = resolve_report_output(
        args.analysis_output,
        disagreement_output,
        suffix=".md",
    )
    analysis_jsonl_output = resolve_report_output(
        args.analysis_jsonl_output,
        disagreement_output,
        suffix=".jsonl",
    )

    preflight_retry_output(retry_a, overwrite=args.overwrite, reuse=args.reuse_retry)
    preflight_retry_output(retry_b, overwrite=args.overwrite, reuse=args.reuse_retry)
    ensure_can_write(disagreement_output, overwrite=args.overwrite)
    ensure_can_write(analysis_output, overwrite=args.overwrite)
    ensure_can_write(analysis_jsonl_output, overwrite=args.overwrite)

    run_retry(
        label="run-a",
        run_dir=run_a,
        output_dir=retry_a,
        retry_config=retry_config,
        model=model_a,
        base_url=args.base_url_a or args.base_url,
        api_key=args.api_key_a or args.api_key,
        metrics=args.metric or [],
        error_types=args.error_types or [],
        input_paths=args.input or [],
        overwrite=args.overwrite,
        reuse=args.reuse_retry,
    )
    run_retry(
        label="run-b",
        run_dir=run_b,
        output_dir=retry_b,
        retry_config=retry_config,
        model=model_b,
        base_url=args.base_url_b or args.base_url,
        api_key=args.api_key_b or args.api_key,
        metrics=args.metric or [],
        error_types=args.error_types or [],
        input_paths=args.input or [],
        overwrite=args.overwrite,
        reuse=args.reuse_retry,
    )
    run_compare(
        run_a=retry_a,
        run_b=retry_b,
        output=disagreement_output,
        fmt=args.format,
        model_a=model_a,
        model_b=model_b,
        metrics=args.metric or [],
        input_paths=args.input or [],
        include_raw=args.include_raw,
    )
    run_analysis(
        input_path=disagreement_output,
        output_path=analysis_output,
        jsonl_output_path=analysis_jsonl_output,
        analyze_with_llm=not args.skip_analysis,
        include_original=args.include_original,
        model=args.analysis_model or os.environ.get("MODEL") or model_b,
        base_url=args.analysis_base_url or args.base_url,
        api_key=args.analysis_api_key or args.api_key,
        max_tokens=args.analysis_max_tokens,
        timeout=args.analysis_timeout,
    )

    print("[pipeline] complete")
    print(f"[pipeline] retry_a: {retry_a}")
    print(f"[pipeline] retry_b: {retry_b}")
    print(f"[pipeline] disagreements: {disagreement_output}")
    print(f"[pipeline] analysis_markdown: {analysis_output}")
    print(f"[pipeline] analysis_jsonl: {analysis_jsonl_output}")


def resolve_path(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else (Path.cwd() / path).resolve()


def resolve_output_dir(raw: str | None, run_dir: Path) -> Path:
    if raw:
        return resolve_path(raw)
    return run_dir.with_name(f"{run_dir.name}_retry")


def resolve_disagreement_output(
    raw: str | None,
    *,
    run_a: Path,
    run_b: Path,
    metrics: list[str],
    fmt: str,
) -> Path:
    if raw:
        return resolve_path(raw)
    root = common_output_root(run_a, run_b)
    label = safe_filename(metrics[0]) if len(metrics) == 1 else "score"
    return root / f"{label}_disagreements_after_retry.{fmt}"


def resolve_report_output(raw: str | None, input_path: Path, *, suffix: str) -> Path:
    if raw:
        return resolve_path(raw)
    return input_path.with_name(f"{input_path.stem}_analysis{suffix}")


def common_output_root(run_a: Path, run_b: Path) -> Path:
    if run_a.parent.parent == run_b.parent.parent:
        return run_a.parent.parent
    if run_a.parent == run_b.parent:
        return run_a.parent
    return run_a.parent


def safe_filename(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value) or "score"


def infer_model_from_run_dir(run_dir: Path) -> str | None:
    value = run_dir.parent.name.strip()
    if not value:
        return None
    if value.endswith("_retry"):
        value = value[: -len("_retry")]
    if "/" in value:
        return value
    if "_" in value:
        provider, rest = value.split("_", 1)
        if provider and rest:
            return f"{provider}/{rest}"
    return value


def preflight_retry_output(output_dir: Path, *, overwrite: bool, reuse: bool) -> None:
    if reuse and (output_dir / "task_results.jsonl").exists():
        return
    existing = [name for name in RETRY_MANAGED_NAMES if (output_dir / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"Retry output already exists: {output_dir}. "
            "Use --overwrite to replace it or --reuse-retry to reuse it."
        )


def ensure_can_write(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {path}. Use --overwrite to replace it.")


def run_retry(
    *,
    label: str,
    run_dir: Path,
    output_dir: Path,
    retry_config: Path,
    model: str | None,
    base_url: str | None,
    api_key: str | None,
    metrics: list[str],
    error_types: list[str],
    input_paths: list[str],
    overwrite: bool,
    reuse: bool,
) -> None:
    if reuse and (output_dir / "task_results.jsonl").exists():
        print(f"[pipeline] {label}: reuse retry output {output_dir}")
        return

    argv = [
        "--run-dir",
        str(run_dir),
        "--retry-config",
        str(retry_config),
        "--output-dir",
        str(output_dir),
    ]
    for metric in metrics:
        argv.extend(["--metric", metric])
    if error_types:
        argv.append("--error-types")
        argv.extend(error_types)
    if input_paths:
        argv.append("--input")
        argv.extend(input_paths)
    if overwrite:
        argv.append("--overwrite")

    print(f"[pipeline] {label}: retry failed results")
    with patched_env({"MODEL": model, "BASE_URL": base_url, "API_KEY": api_key}):
        retry_failed_results.main(argv)


def run_compare(
    *,
    run_a: Path,
    run_b: Path,
    output: Path,
    fmt: str,
    model_a: str | None,
    model_b: str | None,
    metrics: list[str],
    input_paths: list[str],
    include_raw: bool,
) -> None:
    argv = [
        "--run-a",
        str(run_a),
        "--run-b",
        str(run_b),
        "--output",
        str(output),
        "--format",
        fmt,
    ]
    if model_a:
        argv.extend(["--model-a", model_a])
    if model_b:
        argv.extend(["--model-b", model_b])
    for metric in metrics:
        argv.extend(["--metric", metric])
    if input_paths:
        argv.append("--input")
        argv.extend(input_paths)
    if include_raw:
        argv.append("--include-raw")

    print("[pipeline] compare score disagreements")
    extract_score_disagreements.main(argv)


def run_analysis(
    *,
    input_path: Path,
    output_path: Path,
    jsonl_output_path: Path,
    analyze_with_llm: bool,
    include_original: bool,
    model: str | None,
    base_url: str | None,
    api_key: str | None,
    max_tokens: int,
    timeout: float,
) -> None:
    argv = [
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--jsonl-output",
        str(jsonl_output_path),
        "--analysis-max-tokens",
        str(max_tokens),
        "--analysis-timeout",
        str(timeout),
    ]
    if include_original:
        argv.append("--include-original")
    else:
        argv.append("--no-original")
    if analyze_with_llm:
        argv.append("--analyze-with-llm")
        if model:
            argv.extend(["--analysis-model", model])
        if base_url:
            argv.extend(["--analysis-base-url", base_url])
        if api_key:
            argv.extend(["--analysis-api-key", api_key])

    print("[pipeline] write disagreement analysis")
    with patched_env({"MODEL": model, "BASE_URL": base_url, "API_KEY": api_key}):
        summarize_score_disagreements.main(argv)


@contextmanager
def patched_env(values: dict[str, str | None]) -> Iterator[None]:
    previous = {key: os.environ.get(key) for key in values}
    try:
        for key, value in values.items():
            if value is not None:
                os.environ[key] = value
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


if __name__ == "__main__":
    main()

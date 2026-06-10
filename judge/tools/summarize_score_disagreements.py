#!/usr/bin/env python3
"""Summarize score-disagreement JSONL files into grouped reports."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


SCRIPT_PATH = Path(__file__).resolve()
TOOLS_ROOT = SCRIPT_PATH.parent
JUDGE_ROOT = TOOLS_ROOT.parent
REPO_ROOT = JUDGE_ROOT.parent

sys.path.insert(0, str(JUDGE_ROOT))

from core.llm_client import OpenAIChatClient  # noqa: E402
from core.parsing import parse_json_object  # noqa: E402


PAIR_ORDER = ("0_vs_2", "1_vs_2", "0_vs_1", "other")
PAIR_TITLES = {
    "0_vs_2": "0 分 vs 2 分",
    "1_vs_2": "1 分 vs 2 分",
    "0_vs_1": "0 分 vs 1 分",
    "other": "其他分歧",
}
ORIGINAL_FIELDS = ("query", "conversations", "tools", "metadata", "source_path", "raw")


@dataclass
class DisagreementItem:
    sample_id: str
    metric: str
    task_id: str
    score_pair: str
    scores: dict[str, Any]
    results: list[dict[str, Any]]
    original_data: dict[str, Any]
    source_row: dict[str, Any]
    disagreement_summary: str = "未自动分析"
    category: str = "未自动分类"
    analysis_error: dict[str, Any] | None = None


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Read score-disagreement JSONL/JSON, group cases by score pair, "
            "and write a Markdown report plus a structured JSONL summary."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="Disagreement JSONL/JSON file.")
    parser.add_argument(
        "--output",
        default=None,
        help="Markdown output path. Defaults to <input-stem>_analysis.md in the input directory.",
    )
    parser.add_argument(
        "--jsonl-output",
        default=None,
        help="Structured JSONL output path. Defaults to <input-stem>_analysis.jsonl.",
    )
    original_group = parser.add_mutually_exclusive_group()
    original_group.add_argument(
        "--include-original",
        action="store_true",
        help="Include original data in the Markdown report.",
    )
    original_group.add_argument(
        "--no-original",
        dest="include_original",
        action="store_false",
        help="Do not include original data in the Markdown report.",
    )
    parser.set_defaults(include_original=True)
    parser.add_argument(
        "--analyze-with-llm",
        action="store_true",
        help="Use an OpenAI-compatible API to generate disagreement summaries and categories.",
    )
    parser.add_argument(
        "--analysis-model",
        default=None,
        help="Override analysis MODEL. Defaults to env MODEL.",
    )
    parser.add_argument(
        "--analysis-base-url",
        default=None,
        help="Override analysis BASE_URL. Defaults to env BASE_URL.",
    )
    parser.add_argument(
        "--analysis-api-key",
        default=None,
        help="Override analysis API_KEY. Defaults to env API_KEY.",
    )
    parser.add_argument(
        "--analysis-max-tokens",
        type=int,
        default=512,
        help="Max output tokens per LLM analysis request.",
    )
    parser.add_argument(
        "--analysis-timeout",
        type=float,
        default=120.0,
        help="Timeout for each LLM analysis request.",
    )
    args = parser.parse_args(argv)

    input_path = resolve_path(args.input)
    output_path = resolve_output_path(args.output, input_path, suffix=".md")
    jsonl_output_path = resolve_output_path(args.jsonl_output, input_path, suffix=".jsonl")

    rows = read_rows(input_path)
    items = list(expand_disagreements(rows))
    if args.analyze_with_llm and items:
        analyze_items_with_llm(
            items,
            model=args.analysis_model,
            base_url=args.analysis_base_url,
            api_key=args.analysis_api_key,
            max_tokens=args.analysis_max_tokens,
            timeout=args.analysis_timeout,
        )

    write_structured_jsonl(jsonl_output_path, items)
    write_markdown_report(
        output_path,
        items,
        input_path=input_path,
        include_original=args.include_original,
    )

    counts = pair_counts(items)
    print(f"[summary] input: {input_path}")
    print(f"[summary] items: {len(items)}")
    for pair in PAIR_ORDER:
        print(f"[summary] {pair}: {counts.get(pair, 0)}")
    print(f"[summary] markdown: {output_path}")
    print(f"[summary] jsonl: {jsonl_output_path}")


def resolve_path(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else (Path.cwd() / path).resolve()


def resolve_output_path(raw: str | None, input_path: Path, *, suffix: str) -> Path:
    if raw:
        return resolve_path(raw)
    return input_path.with_name(f"{input_path.stem}_analysis{suffix}")


def read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError(f"JSON input must be a list: {path}")
        return data
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        raw = line.strip()
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
        if not isinstance(obj, dict):
            raise ValueError(f"JSONL row must be an object at {path}:{line_number}")
        rows.append(obj)
    return rows


def expand_disagreements(rows: Iterable[dict[str, Any]]) -> Iterable[DisagreementItem]:
    for row in rows:
        sample_id = str(row.get("sample_id") or "")
        if not sample_id:
            continue
        disagreements = row.get("judge_disagreements")
        if not isinstance(disagreements, list) or not disagreements:
            single = row.get("judge_disagreement")
            disagreements = [single] if isinstance(single, dict) else []
        for disagreement in disagreements:
            if not isinstance(disagreement, dict):
                continue
            metric = str(disagreement.get("metric") or "")
            task_id = str(disagreement.get("task_id") or "")
            scores = disagreement.get("scores") or {}
            if not metric or not task_id or not isinstance(scores, dict):
                continue
            results = matching_results(row.get("judge_results") or [], metric, task_id)
            yield DisagreementItem(
                sample_id=sample_id,
                metric=metric,
                task_id=task_id,
                score_pair=score_pair_label(scores.values()),
                scores=dict(scores),
                results=results,
                original_data=extract_original_data(row),
                source_row=row,
            )


def matching_results(results: Any, metric: str, task_id: str) -> list[dict[str, Any]]:
    if not isinstance(results, list):
        return []
    matched = [
        result
        for result in results
        if isinstance(result, dict)
        and str(result.get("metric") or "") == metric
        and str(result.get("task_id") or "") == task_id
    ]
    return matched or [result for result in results if isinstance(result, dict)]


def extract_original_data(row: dict[str, Any]) -> dict[str, Any]:
    original = {field: row[field] for field in ORIGINAL_FIELDS if field in row}
    if "metadata" not in original:
        original["metadata"] = {}
    return original


def score_pair_label(values: Iterable[Any]) -> str:
    normalized = [normalize_score(value) for value in values]
    score_set: set[int] = set()
    for value in normalized:
        if value in {0, 1, 2}:
            score_set.add(value)
        else:
            return "other"
    if score_set == {0, 2}:
        return "0_vs_2"
    if score_set == {1, 2}:
        return "1_vs_2"
    if score_set == {0, 1}:
        return "0_vs_1"
    return "other"


def normalize_score(value: Any) -> Any:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    if number.is_integer():
        return int(number)
    return number


def pair_counts(items: Iterable[DisagreementItem]) -> dict[str, int]:
    counts = {pair: 0 for pair in PAIR_ORDER}
    for item in items:
        counts[item.score_pair] = counts.get(item.score_pair, 0) + 1
    return counts


def analyze_items_with_llm(
    items: list[DisagreementItem],
    *,
    model: str | None,
    base_url: str | None,
    api_key: str | None,
    max_tokens: int,
    timeout: float,
) -> None:
    model = model or os.environ.get("MODEL")
    base_url = base_url or os.environ.get("BASE_URL")
    api_key = api_key or os.environ.get("API_KEY")
    client = OpenAIChatClient(
        model=model,
        base_url=base_url,
        api_key=api_key,
        timeout=timeout,
        max_retries=2,
        retry_delay=1.0,
    )
    for index, item in enumerate(items, start=1):
        payload = {
            "messages": build_analysis_messages(item),
            "temperature": 0.0,
            "max_tokens": max_tokens,
        }
        response = client(payload, max_concurrency=1)
        if not response.ok:
            item.analysis_error = response.error or {"type": "llm_error"}
            print(
                f"[analysis] {index}/{len(items)} failed: {item.sample_id} {item.analysis_error}",
                file=sys.stderr,
            )
            continue
        try:
            parsed = parse_json_object(response.text or "")
            summary = parsed.get("disagreement_summary")
            category = parsed.get("category")
            if not isinstance(summary, str) or not summary.strip():
                raise ValueError("Missing disagreement_summary")
            if not isinstance(category, str) or not category.strip():
                raise ValueError("Missing category")
            item.disagreement_summary = summary.strip()
            item.category = category.strip()
        except Exception as exc:  # noqa: BLE001 - keep report generation robust
            item.analysis_error = {
                "type": type(exc).__name__,
                "message": str(exc),
                "raw_text": (response.text or "")[:1000],
            }
        print(f"[analysis] {index}/{len(items)} {item.sample_id}")


def build_analysis_messages(item: DisagreementItem) -> list[dict[str, str]]:
    system = (
        "你是数据质量打分分歧分析助手。你只输出严格 JSON object，"
        "不要输出 Markdown、解释过程或 JSON 之外的文本。"
    )
    user = (
        "请根据原始数据和两个 judge 模型的打分理由，总结分歧点和分歧大类。\n"
        "要求：\n"
        "1. disagreement_summary 用一句话说明两个模型到底分歧在哪里。\n"
        "2. category 用一个短语概括分歧大类，例如“对象是否足够明确边界”、"
        "“相对时间/日期转换边界”、“工具选择问题泄漏到该指标”、"
        "“评分严重程度边界”。\n"
        "3. 不要判断哪个模型一定正确，只归纳分歧。\n"
        "4. 只输出 JSON，格式为："
        "{\"disagreement_summary\":\"...\",\"category\":\"...\"}\n\n"
        f"sample_id: {item.sample_id}\n"
        f"metric: {item.metric}\n"
        f"task_id: {item.task_id}\n"
        f"scores:\n{json.dumps(item.scores, ensure_ascii=False, indent=2)}\n\n"
        f"judge_results:\n{compact_json(item.results, max_chars=7000)}\n\n"
        f"original_data:\n{compact_json(item.original_data, max_chars=9000)}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def compact_json(value: Any, *, max_chars: int) -> str:
    text = json.dumps(value, ensure_ascii=False, indent=2)
    if len(text) <= max_chars:
        return text
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
    return f"{text[:max_chars]}\n... [truncated, sha1={digest}, original_chars={len(text)}]"


def write_structured_jsonl(path: Path, items: list[DisagreementItem]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item_to_dict(item), ensure_ascii=False) + "\n")


def item_to_dict(item: DisagreementItem) -> dict[str, Any]:
    return {
        "sample_id": item.sample_id,
        "metric": item.metric,
        "task_id": item.task_id,
        "score_pair": item.score_pair,
        "scores": item.scores,
        "judge_results": item.results,
        "disagreement_summary": item.disagreement_summary,
        "category": item.category,
        "analysis_error": item.analysis_error,
        "original_data": item.original_data,
    }


def write_markdown_report(
    path: Path,
    items: list[DisagreementItem],
    *,
    input_path: Path,
    include_original: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    grouped = {pair: [item for item in items if item.score_pair == pair] for pair in PAIR_ORDER}
    counts = pair_counts(items)
    lines: list[str] = [
        "# Score Disagreement Analysis",
        "",
        f"来源文件: `{input_path}`",
        f"总分歧数: {len(items)}",
        "",
        "## 汇总",
        "",
        "| 分数组合 | 数量 |",
        "|---|---:|",
    ]
    for pair in PAIR_ORDER:
        if pair == "other" and counts.get(pair, 0) == 0:
            continue
        lines.append(f"| {PAIR_TITLES[pair].replace(' 分', '')} | {counts.get(pair, 0)} |")
    lines.append("")

    for pair in PAIR_ORDER:
        section_items = grouped.get(pair) or []
        if not section_items:
            continue
        lines.extend([f"## {PAIR_TITLES[pair]}", ""])
        for index, item in enumerate(section_items, start=1):
            lines.extend(item_markdown(index, item, include_original=include_original))
            lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def item_markdown(
    index: int,
    item: DisagreementItem,
    *,
    include_original: bool,
) -> list[str]:
    lines = [
        f"### {index}. {item.sample_id}",
        f"- 指标: {item.metric}",
        f"- 分数: {format_scores(item.scores)}",
    ]
    for result in item.results:
        model = str(result.get("model") or "model")
        lines.append(f"- {display_model_name(model)} 判断: {single_line(result_reason(result))}")
    lines.extend([
        f"- 分歧点: {single_line(item.disagreement_summary)}",
        f"- 我判断的大类: {single_line(item.category)}",
    ])
    if item.analysis_error:
        lines.append(f"- 自动分析错误: `{json.dumps(item.analysis_error, ensure_ascii=False)}`")
    if include_original:
        lines.extend([
            "",
            "<details>",
            "<summary>查看原始数据</summary>",
            "",
            "```json",
            json.dumps(item.original_data, ensure_ascii=False, indent=2),
            "```",
            "",
            "</details>",
        ])
    return lines


def format_scores(scores: dict[str, Any]) -> str:
    parts = []
    for model, score in scores.items():
        parts.append(f"{display_model_name(str(model))}={score}")
    return "，".join(parts)


def display_model_name(model: str) -> str:
    lower = model.lower()
    if "deepseek" in lower:
        return "DeepSeek"
    if "gpt" in lower or "openai" in lower:
        return "GPT"
    return model


def result_reason(result: dict[str, Any]) -> str:
    reason = result.get("reason")
    if isinstance(reason, str) and reason.strip():
        return reason
    details = result.get("details") or {}
    if isinstance(details, dict):
        explanation = details.get("explanation")
        if isinstance(explanation, str) and explanation.strip():
            return explanation
    error = result.get("error")
    if error:
        return json.dumps(error, ensure_ascii=False)
    return ""


def single_line(value: str) -> str:
    return " ".join(str(value).split())


if __name__ == "__main__":
    main()

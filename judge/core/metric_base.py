"""Reusable base classes for sample-level JSON judge metrics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .parsing import parse_json_object
from .prompts import render_template
from .registry import BaseMetric
from .schemas import JudgeCase, JudgeResult, JudgeTask


EVIDENCE_SOURCE_TYPES = [
    "query",
    "history",
    "conversations",
    "assistant",
    "tool_schema",
    "tools",
    "tool_call",
    "tool_response",
    "final_answer",
]


class SampleJsonMetric(BaseMetric):
    """Sample-level metric whose judge response is a JSON object."""

    allowed_issue_types: set[str] = set()
    required_fields: set[str] = {"score", "issue_types", "explanation"}
    output_properties: dict[str, Any] = {}

    def build_tasks(self, sample: JudgeCase) -> list[JudgeTask]:
        metric_dir = Path(self.config.path)
        prompt_name = self.config.prompt or "prompt.j2"
        prompt_path = metric_dir / prompt_name
        prompt = render_template(prompt_path, self._template_values(sample))
        return [
            JudgeTask(
                metric=self.config.name,
                metric_version=self.config.version,
                sample_id=sample.sample_id,
                task_id="sample",
                messages=self._build_messages(prompt),
                prompt=prompt,
                params=self._request_params(),
                payload_meta={
                    "task_granularity": "sample",
                    "query": sample.query,
                    "source_index": sample.source_index,
                    "source_path": sample.source_path,
                    "source_row": sample.source_row,
                },
            )
        ]

    def _build_messages(self, prompt: str) -> list[dict[str, str]]:
        system = (
            "你是一个数据质量评估器。你必须只输出一个可被 json.loads "
            "解析的 JSON object。不要输出 Markdown、代码块、解释、分析过程、"
            "前缀或后缀。如果需要思考，只在内部完成，不要写出。输出的第一个"
            "字符必须是 {，最后一个字符必须是 }。"
        )
        user = (
            f"{prompt}\n\n"
            "现在只返回最终 JSON object。不要复述任务，不要解释，不要输出任何 JSON 之外的文本。\n"
            "/no_think"
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def parse_response(
        self,
        task: JudgeTask,
        response: dict[str, Any],
    ) -> JudgeResult:
        if not response.get("ok", False):
            return JudgeResult(
                sample_id=task.sample_id,
                metric=task.metric,
                metric_version=task.metric_version,
                task_id=task.task_id,
                error=response.get("error") or {"type": "llm_error"},
            )

        text = response.get("text") or ""
        try:
            parsed = parse_json_object(text)
            self._validate_common(parsed)
        except Exception as exc:  # noqa: BLE001 - surface parse/validation errors
            return JudgeResult(
                sample_id=task.sample_id,
                metric=task.metric,
                metric_version=task.metric_version,
                task_id=task.task_id,
                score=None,
                reason="LLM output is not valid metric JSON.",
                details={"raw_text": text[:4000]},
                error={
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
            )

        return JudgeResult(
            sample_id=task.sample_id,
            metric=task.metric,
            metric_version=task.metric_version,
            task_id=task.task_id,
            score=float(parsed["score"]),
            passed=parsed["score"] >= 2,
            reason=parsed.get("explanation") or parsed.get("reason"),
            details=parsed,
            error=None,
        )

    def _template_values(self, sample: JudgeCase) -> dict[str, Any]:
        return {
            "query": sample.query or _first_user_content(sample.conversations),
            "conversations_json": json.dumps(
                sample.conversations,
                ensure_ascii=False,
                indent=2,
            ),
            "tools_json": json.dumps(sample.tools, ensure_ascii=False, indent=2),
            "metadata_json": json.dumps(
                sample.metadata or {},
                ensure_ascii=False,
                indent=2,
            ),
            "raw_json": json.dumps(sample.raw or {}, ensure_ascii=False, indent=2),
            "allowed_issue_types": json.dumps(
                sorted(self.allowed_issue_types),
                ensure_ascii=False,
            ),
            "output_schema_json": json.dumps(
                self._output_schema(),
                ensure_ascii=False,
                indent=2,
            ),
        }

    def _request_params(self) -> dict[str, Any]:
        allowed = {
            "temperature",
            "max_tokens",
            "max_new_tokens",
            "return_logprob",
            "top_logprobs_num",
        }
        return {k: v for k, v in self.config.params.items() if k in allowed}

    def _validate_common(self, parsed: dict[str, Any]) -> None:
        missing = self.required_fields - set(parsed)
        if missing:
            raise ValueError(f"Missing required fields: {sorted(missing)}")

        score = parsed.get("score")
        if not isinstance(score, int) or isinstance(score, bool) or score not in {0, 1, 2}:
            raise ValueError("score must be one of integer 0, 1, 2")

        issue_types = parsed.get("issue_types")
        if not isinstance(issue_types, list) or not all(
            isinstance(x, str) for x in issue_types
        ):
            raise ValueError("issue_types must be a list of strings")
        unknown = set(issue_types) - self.allowed_issue_types
        if unknown:
            raise ValueError(f"Unknown issue_types: {sorted(unknown)}")

        affected_turns = parsed.get("affected_turns", [])
        if not isinstance(affected_turns, list) or not all(
            isinstance(x, int) and not isinstance(x, bool) for x in affected_turns
        ):
            raise ValueError("affected_turns must be a list of integer indexes")

        confidence = parsed.get("confidence")
        if confidence is not None and not isinstance(confidence, (int, float)):
            raise ValueError("confidence must be numeric when provided")

    def _output_schema(self) -> dict[str, Any]:
        properties: dict[str, Any] = {
            "score": {
                "type": "integer",
                "enum": [0, 1, 2],
                "description": "0=存在明显问题，不采用；1=稍欠合理性，考虑保留；2=合理，保留",
            },
            "issue_types": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": sorted(self.allowed_issue_types),
                },
                "description": "只能使用指标定义中列出的问题类型；无问题时返回空数组",
            },
            "affected_turns": {
                "type": "array",
                "items": {"type": "integer"},
                "description": "相关 conversations 下标；无明确定位时返回空数组",
            },
            "explanation": {
                "type": "string",
                "description": "一句到三句话说明评分依据",
            },
            "evidence": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "source": {
                            "type": "string",
                            "enum": EVIDENCE_SOURCE_TYPES,
                        },
                        "quote": {"type": "string"},
                    },
                    "required": ["source", "quote"],
                    "additionalProperties": False,
                },
            },
            "confidence": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
            },
        }
        properties.update(self.output_properties)
        return {
            "type": "object",
            "properties": properties,
            "required": sorted(self.required_fields),
            "additionalProperties": False,
        }


def _first_user_content(conversations: list[dict[str, Any]]) -> str | None:
    for msg in conversations:
        if msg.get("role") == "user":
            content = msg.get("content")
            return str(content) if content is not None else None
    return None

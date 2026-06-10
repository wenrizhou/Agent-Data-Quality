from __future__ import annotations

from typing import Any

from core.metric_base import SampleJsonMetric


class Metric(SampleJsonMetric):
    required_fields = {
        "score",
        "reason",
        "unmet_requirements",
        "requirement_analysis",
    }

    def _validate_common(self, parsed: dict[str, Any]) -> None:
        missing = self.required_fields - set(parsed)
        if missing:
            raise ValueError(f"Missing required fields: {sorted(missing)}")

        score = parsed.get("score")
        if not isinstance(score, int) or isinstance(score, bool) or score not in {0, 1, 2}:
            raise ValueError("score must be one of integer 0, 1, 2")

        reason = parsed.get("reason")
        if not isinstance(reason, str):
            raise ValueError("reason must be a string")

        unmet_requirements = parsed.get("unmet_requirements")
        if not isinstance(unmet_requirements, list) or not all(
            isinstance(x, str) for x in unmet_requirements
        ):
            raise ValueError("unmet_requirements must be a list of strings")

        requirement_analysis = parsed.get("requirement_analysis")
        if not isinstance(requirement_analysis, str):
            raise ValueError("requirement_analysis must be a string")

    def _output_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "score": {
                    "type": "integer",
                    "enum": [0, 1, 2],
                    "description": "0=未满足核心需求；1=部分满足；2=充分满足全部核心需求",
                },
                "reason": {
                    "type": "string",
                    "description": "简要说明为什么给这个分数",
                },
                "unmet_requirements": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "列出未被满足或只被部分满足的用户需求；无则返回空数组",
                },
                "requirement_analysis": {
                    "type": "string",
                    "description": "简要列出识别到的主要用户需求",
                },
            },
            "required": sorted(self.required_fields),
            "additionalProperties": False,
        }

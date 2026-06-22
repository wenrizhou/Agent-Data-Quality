"""English prompt wrappers for sample-level JSON judge metrics."""

from __future__ import annotations

from typing import Any

from .metric_base import SampleJsonMetric


class EnglishSampleJsonMetric(SampleJsonMetric):
    """Sample JSON metric with English chat wrappers and schema descriptions."""

    def _build_messages(self, prompt: str) -> list[dict[str, str]]:
        system = (
            "You are a data quality evaluator. You must output only one JSON object "
            "that can be parsed by json.loads. Do not output Markdown, code fences, "
            "explanations, analysis, prefixes, or suffixes. If you need to think, "
            "do it internally and do not write it out. The first character of the "
            "output must be {, and the last character must be }."
        )
        user = (
            f"{prompt}\n\n"
            "Now return only the final JSON object. Do not restate the task, "
            "do not explain, and do not output any text outside the JSON object.\n"
            "/no_think"
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def _output_schema(self) -> dict[str, Any]:
        schema = super()._output_schema()
        properties = schema["properties"]

        properties["score"]["description"] = (
            "0=core requirement clearly unmet; "
            "1=partially fulfilled or minor reasonableness issue; "
            "2=reasonable and fully fulfilled"
        )
        properties["issue_types"]["description"] = (
            "Use only issue types listed in the metric definition; return an empty "
            "array when there is no issue."
        )
        properties["affected_turns"]["description"] = (
            "Relevant conversation indexes; return an empty array when not "
            "applicable or hard to locate."
        )
        properties["explanation"]["description"] = (
            "Explain the scoring basis in one to three sentences."
        )
        return schema

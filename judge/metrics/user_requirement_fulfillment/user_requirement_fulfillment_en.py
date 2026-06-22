from __future__ import annotations

from typing import Any

from core.metric_base_en import EnglishSampleJsonMetric


class Metric(EnglishSampleJsonMetric):
    allowed_issue_types = {
        "no_user_visible_result",
        "unnecessary_clarification",
        "wrong_or_unsupported_execution_path",
        "incomplete_multistep_workflow",
        "missing_explicit_requirement",
        "weak_or_generic_limitation_handling",
        "unsupported_or_hallucinated_content",
        "format_or_constraint_violation",
        "missing_confirmation_or_artifact",
        "off_target_or_wrong_task",
    }
    required_fields = {
        "score",
        "issue_types",
        "affected_turns",
        "explanation",
        "unmet_requirements",
        "requirement_analysis",
        "evidence",
        "confidence",
    }
    output_properties = {
        "unmet_requirements": {
            "type": "array",
            "items": {"type": "string"},
            "description": "User requirements that are unmet or only partially met; empty if none.",
        },
        "requirement_analysis": {
            "type": "string",
            "description": "Brief summary of the currently active main user requirements.",
        },
    }

    def _validate_common(self, parsed: dict[str, Any]) -> None:
        super()._validate_common(parsed)

        unmet_requirements = parsed.get("unmet_requirements")
        if not isinstance(unmet_requirements, list) or not all(
            isinstance(x, str) for x in unmet_requirements
        ):
            raise ValueError("unmet_requirements must be a list of strings")

        requirement_analysis = parsed.get("requirement_analysis")
        if not isinstance(requirement_analysis, str):
            raise ValueError("requirement_analysis must be a string")

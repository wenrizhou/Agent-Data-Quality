from __future__ import annotations

from typing import Any

from core.metric_base import SampleJsonMetric


class Metric(SampleJsonMetric):
    required_fields = {
        "score",
        "reason",
        "unsupported_or_contradicted_claims",
        "evidence_basis",
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

        claims = parsed.get("unsupported_or_contradicted_claims")
        if not isinstance(claims, list) or not all(isinstance(x, str) for x in claims):
            raise ValueError("unsupported_or_contradicted_claims must be a list of strings")

        evidence_basis = parsed.get("evidence_basis")
        if not isinstance(evidence_basis, str):
            raise ValueError("evidence_basis must be a string")

    def _output_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "score": {
                    "type": "integer",
                    "enum": [0, 1, 2],
                    "description": (
                        "0=poor evidence support; 1=partially supported but has "
                        "meaningful issues; 2=all key content is evidence-supported"
                    ),
                },
                "reason": {
                    "type": "string",
                    "description": "Briefly explain why this score was assigned.",
                },
                "unsupported_or_contradicted_claims": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "List key final-response claims that are unsupported, "
                        "unverifiable, overextended, or contradicted by evidence; "
                        "return an empty array when none exist."
                    ),
                },
                "evidence_basis": {
                    "type": "string",
                    "description": (
                        "Briefly state the main user, context, tool, metadata, "
                        "or raw evidence used for the judgment."
                    ),
                },
            },
            "required": sorted(self.required_fields),
            "additionalProperties": False,
        }

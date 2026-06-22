from core.metric_base import SampleJsonMetric


class Metric(SampleJsonMetric):
    allowed_issue_types = {
        "missing_necessary_tool",
        "tool_capability_mismatch",
        "irrelevant_tool_call",
        "incorrect_tool_chain_order",
    }
    required_fields = {
        "score",
        "affected_turns",
        "checked_tool_calls",
        "missing_tools",
        "issue_types",
        "explanation",
        "evidence",
        "confidence",
    }
    output_properties = {
        "checked_tool_calls": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "tool_call_id": {"type": "string"},
                    "tool_name": {"type": "string"},
                    "score": {"type": "integer", "enum": [0, 1, 2]},
                    "issue_types": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": sorted(allowed_issue_types),
                        },
                    },
                    "reason": {"type": "string"},
                },
                "required": [
                    "tool_call_id",
                    "tool_name",
                    "score",
                    "issue_types",
                    "reason",
                ],
                "additionalProperties": False,
            },
        },
        "missing_tools": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "requirement": {"type": "string"},
                    "expected_tool_or_capability": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": [
                    "requirement",
                    "expected_tool_or_capability",
                    "reason",
                ],
                "additionalProperties": False,
            },
        },
    }

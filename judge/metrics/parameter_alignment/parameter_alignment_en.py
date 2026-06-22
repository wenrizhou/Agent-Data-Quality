from core.metric_base import SampleJsonMetric


class Metric(SampleJsonMetric):
    allowed_issue_types = {
        "invalid_parameter_value",
        "user_parameter_extraction_error",
        "trajectory_parameter_extraction_error",
    }
    required_fields = {
        "score",
        "affected_turns",
        "checked_tool_calls",
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
                    "invalid_parameters": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "value": {},
                                "issue_type": {
                                    "type": "string",
                                    "enum": sorted(allowed_issue_types),
                                },
                                "reason": {"type": "string"},
                            },
                            "required": ["name", "value", "issue_type", "reason"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": [
                    "tool_call_id",
                    "tool_name",
                    "score",
                    "invalid_parameters",
                ],
                "additionalProperties": False,
            },
        },
    }

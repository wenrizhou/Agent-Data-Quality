from core.metric_base import SampleJsonMetric


class Metric(SampleJsonMetric):
    allowed_issue_types = {
        "参数值不合法",
        "用户参数提取错误",
        "轨迹参数提取错误",
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

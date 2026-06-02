from core.metric_base import SampleJsonMetric


class Metric(SampleJsonMetric):
    allowed_issue_types = {
        "冗余重复",
        "无结果引用",
    }
    required_fields = {
        "score",
        "necessary_tool_calls",
        "redundant_tool_calls",
        "unused_tool_responses",
        "final_answer_uses_any_tool_response",
        "affected_turns",
        "issue_types",
        "explanation",
        "evidence",
        "confidence",
    }
    output_properties = {
        "necessary_tool_calls": {
            "type": "array",
            "items": {"type": "string"},
        },
        "redundant_tool_calls": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "tool_call_id": {"type": "string"},
                    "tool_name": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["tool_call_id", "tool_name", "reason"],
                "additionalProperties": False,
            },
        },
        "unused_tool_responses": {
            "type": "array",
            "items": {"type": "string"},
        },
        "final_answer_uses_any_tool_response": {"type": "boolean"},
    }

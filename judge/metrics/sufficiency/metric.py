from core.metric_base import SampleJsonMetric


class Metric(SampleJsonMetric):
    allowed_issue_types = {
        "解答错误",
        "轨迹不合理",
        "格式错误",
        "权限不足",
    }
    required_fields = {
        "score",
        "covered_requirements",
        "missing_requirements",
        "final_answer_uses_tool_results",
        "format_followed",
        "blocked_by_permission",
        "affected_turns",
        "issue_types",
        "explanation",
        "evidence",
        "confidence",
    }
    output_properties = {
        "covered_requirements": {
            "type": "array",
            "items": {"type": "string"},
        },
        "missing_requirements": {
            "type": "array",
            "items": {"type": "string"},
        },
        "final_answer_uses_tool_results": {"type": "boolean"},
        "format_followed": {"type": "boolean"},
        "blocked_by_permission": {"type": "boolean"},
    }

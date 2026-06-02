from core.metric_base import SampleJsonMetric


class Metric(SampleJsonMetric):
    allowed_issue_types = {
        "语义割裂",
        "步骤不合理",
    }
    required_fields = {
        "score",
        "affected_turns",
        "issue_types",
        "conflicting_spans",
        "explanation",
        "evidence",
        "confidence",
    }
    output_properties = {
        "conflicting_spans": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "span": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["span", "reason"],
                "additionalProperties": False,
            },
        },
    }

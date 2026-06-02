from core.metric_base import SampleJsonMetric


class Metric(SampleJsonMetric):
    allowed_issue_types = {
        "指向性不可解",
        "整体不可解",
    }
    required_fields = {
        "score",
        "solvable",
        "matched_tools",
        "missing_capabilities",
        "affected_turns",
        "issue_types",
        "explanation",
        "evidence",
        "confidence",
    }
    output_properties = {
        "solvable": {"type": "boolean"},
        "matched_tools": {
            "type": "array",
            "items": {"type": "string"},
        },
        "missing_capabilities": {
            "type": "array",
            "items": {"type": "string"},
        },
    }

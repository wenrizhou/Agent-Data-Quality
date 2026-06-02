from core.metric_base import SampleJsonMetric


class Metric(SampleJsonMetric):
    allowed_issue_types = {
        "意图不清",
        "意图缺失",
        "对象不清",
        "对象缺失",
    }
    required_fields = {
        "score",
        "action_specificity",
        "object_specificity",
        "intent_missing",
        "object_missing",
        "context_resolved",
        "affected_turns",
        "issue_types",
        "explanation",
        "evidence",
        "confidence",
    }
    output_properties = {
        "action_specificity": {"type": "integer", "enum": [0, 1, 2]},
        "object_specificity": {"type": "integer", "enum": [0, 1, 2]},
        "intent_missing": {"type": "boolean"},
        "object_missing": {"type": "boolean"},
        "context_resolved": {"type": "boolean"},
    }

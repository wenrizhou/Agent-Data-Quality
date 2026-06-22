# Export Sample Metric Results

`export_sample_metric_results.py` converts a judge run's `task_results.jsonl`
from one row per `(sample_id, metric, task_id)` into one row per `sample_id`.

It is intended for building quality-scoring training data where each sample
keeps all source metric results and receives a derived `data_quality` score.

## Input

The tool reads a completed judge run directory containing:

```text
task_results.jsonl
run_config.yaml
```

`run_config.yaml` is optional but recommended. When present, the tool uses its
`metrics:` list as the expected source metrics. If no run config is found, it
defaults to the English all-metrics set:

```text
specificity_en
parameter_alignment_en
tool_selection_en
user_requirement_fulfillment_en
final_response_evidence_support_en
minimality_en
```

## Output

By default, the tool writes:

```text
<run-dir>/sample_metric_results.jsonl
```

Each line has this shape:

```json
{
  "sample_id": "sample-1",
  "metrics": {
    "specificity_en": {
      "metric": "specificity_en",
      "metric_version": "0.1.0",
      "task_id": "sample",
      "score": 2.0,
      "passed": true,
      "reason": "...",
      "details": {},
      "error": null,
      "raw_response_id": "...",
      "params_hash": "..."
    }
  },
  "data_quality": {
    "metric": "data_quality",
    "score": 1.0,
    "aggregation": "min",
    "source_metrics": [
      "specificity_en",
      "parameter_alignment_en",
      "tool_selection_en",
      "user_requirement_fulfillment_en",
      "final_response_evidence_support_en",
      "minimality_en"
    ],
    "missing_metrics": [],
    "invalid_metrics": []
  }
}
```

`data_quality.score` is the minimum score across all expected source metrics for
that sample. The tool does not generate `data_quality.passed`, because the
training use case needs 0/1/2 labels rather than a binary pass/fail field.

If any expected metric is missing, has `score: null`, has a non-0/1/2 score, or
has a non-null `error`, then:

```json
"data_quality": {
  "score": null,
  "missing_metrics": ["minimality_en"],
  "invalid_metrics": ["parameter_alignment_en"]
}
```

This avoids treating incomplete judge output as a valid training label.

## Usage

Run on a completed judge output directory:

```bash
python judge/tools/export_sample_metric_results.py \
  --run-dir judge/outputs/all_metrics_api_en/qwen3-4b-instruct-2507/<timestamp>
```

Write to a custom path:

```bash
python judge/tools/export_sample_metric_results.py \
  --run-dir judge/outputs/all_metrics_api_en/qwen3-4b-instruct-2507/<timestamp> \
  --output /path/to/sample_metric_results.jsonl
```

Override the expected metrics explicitly:

```bash
python judge/tools/export_sample_metric_results.py \
  --run-dir judge/outputs/all_metrics_api_en/qwen3-4b-instruct-2507/<timestamp> \
  --metric specificity_en \
  --metric parameter_alignment_en \
  --metric tool_selection_en \
  --metric user_requirement_fulfillment_en \
  --metric final_response_evidence_support_en \
  --metric minimality_en
```

## Sampling Labels

After export, use `data_quality.score` to select training examples:

```bash
jq 'select(.data_quality.score == 0)' sample_metric_results.jsonl
jq 'select(.data_quality.score == 1)' sample_metric_results.jsonl
jq 'select(.data_quality.score == 2)' sample_metric_results.jsonl
```

Rows with `data_quality.score == null` should usually be excluded from training
until the missing or invalid metric results are retried.

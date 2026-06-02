# Agent Generate Judge

This directory contains a pluggable LLM-judge runner for canonical agent data.

Current status:

- Canonical JSON, JSONL, and JSONL.GZ loading is implemented.
- Batch config parsing is implemented.
- Metric discovery is implemented.
- The LLM client is implemented locally under `core/llm_client.py`.
- No concrete metrics are included yet.

Run the empty smoke config:

```powershell
python data_quality\judge\run_judge.py --config data_quality\judge\configs\judge_empty.yaml
```

Run all current metric plugins against a live judge backend:

```powershell
python data_quality\judge\run_judge.py --config data_quality\judge\configs\judge_all_metrics.yaml
```

Outputs are timestamped by default when `output.dir: auto`:

```text
data_quality/judge/outputs/<run_name>/<YYYYMMDD_HHMMSS>/
  raw_responses.jsonl
  task_results.jsonl
  summary.json
  run_config.yaml
```

Use a fixed path if you want old resume-in-place behavior:

```yaml
output:
  dir: data_quality/judge/outputs/debug_run
  resume: true
```

Add a future metric as:

```text
data_quality/judge/metrics/<metric_name>/
  metric.yaml
  prompt.j2
  metric.py
```

`metric.py` must define `class Metric`. The class receives `MetricConfig` and
implements:

```python
def build_tasks(self, sample): ...
def parse_response(self, task, response): ...
def aggregate(self, results): ...
```

`metric.yaml` stores metric metadata, defaults, output expectations, and the
default prompt path. `prompt.j2` stores only the prompt template.

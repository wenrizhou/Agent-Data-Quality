"""Judge runner orchestration."""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from .data import expand_input_paths, load_cases
from .io import append_jsonl, load_config, read_jsonl
from .llm_client import JudgeLLMClient, OpenAIChatClient
from .registry import MetricPlugin, load_configured_metrics
from .schemas import JudgeResult, JudgeTask


def run_from_config(
    config_path: Path,
    *,
    input_paths: list[str] | None = None,
    max_samples: int | None = None,
    balanced_sample: bool = False,
    seed: int = 42,
) -> None:
    config = load_config(config_path)
    judge_root = Path(__file__).resolve().parent.parent
    output_dir = _resolve_output_dir(config.get("output") or {}, base_dir=judge_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(config_path, output_dir / f"run_config{config_path.suffix}")
    print(f"[judge] Output dir: {output_dir}")

    input_cfg = config.get("input") or {}
    paths = input_paths if input_paths is not None else input_cfg.get("paths") or []
    sample_limit = max_samples if max_samples is not None else input_cfg.get("max_samples")
    source_files = expand_input_paths(paths)
    cases = load_cases(
        paths,
        max_samples=sample_limit,
        balanced_sample=balanced_sample,
        seed=seed,
    )
    print(f"[judge] Discovered {len(source_files)} input files")
    if balanced_sample:
        print(f"[judge] Balanced sampling enabled: max_samples={sample_limit}, seed={seed}")
    print(f"[judge] Loaded {len(cases)} samples")

    metric_entries = config.get("metrics") or []
    metrics_root = Path(config.get("metrics_root") or judge_root / "metrics")
    if not metric_entries:
        print("[judge] No metrics configured. Nothing to judge.")
        _write_empty_summary(output_dir, len(cases))
        return

    metrics = load_configured_metrics(metrics_root, metric_entries)
    print(f"[judge] Loaded {len(metrics)} metrics: "
          f"{', '.join(m.config.name for m in metrics)}")

    client = _build_client(config.get("client") or {})
    resume = bool((config.get("output") or {}).get("resume", True))
    runner = JudgeRunner(
        metrics=metrics,
        client=client,
        output_dir=output_dir,
        client_config=config.get("client") or {},
        resume=resume,
    )
    runner.run(cases)


class JudgeRunner:
    def __init__(
        self,
        *,
        metrics: list[MetricPlugin],
        client: Any,
        output_dir: Path,
        client_config: dict[str, Any],
        resume: bool,
    ) -> None:
        self.metrics = metrics
        self.client = client
        self.output_dir = output_dir
        self.client_config = client_config
        self.resume = resume
        self.raw_path = output_dir / "raw_responses.jsonl"
        self.result_path = output_dir / "task_results.jsonl"
        self.summary_path = output_dir / "summary.json"

    def run(self, cases: list[Any]) -> None:
        done = self._load_done_keys() if self.resume else set()
        all_results: list[JudgeResult] = []
        if self.resume:
            all_results.extend(self._load_existing_results())

        total_tasks = 0
        skipped = 0
        for metric in self.metrics:
            tasks = []
            for case in cases:
                for task in metric.build_tasks(case):
                    total_tasks += 1
                    if _task_key(task, self.client_config) in done:
                        skipped += 1
                        continue
                    tasks.append(task)
            print(f"[judge] {metric.config.name}: {len(tasks)} pending tasks")
            all_results.extend(self._run_metric(metric, tasks))

        summary = self._build_summary(all_results, total_tasks, skipped)
        self.summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"[judge] Wrote summary: {self.summary_path}")

    def _run_metric(
        self,
        metric: MetricPlugin,
        tasks: list[JudgeTask],
    ) -> list[JudgeResult]:
        if not tasks:
            return []
        batch_size = int(self.client_config.get("batch_size", 128))
        max_concurrency = int(self.client_config.get("max_concurrency", 32))
        results: list[JudgeResult] = []

        for start in range(0, len(tasks), batch_size):
            batch = tasks[start : start + batch_size]
            payloads = [_task_to_payload(t, self.client_config) for t in batch]
            responses = self.client(payloads, max_concurrency=max_concurrency)
            assert isinstance(responses, list)

            for task, response in zip(batch, responses):
                raw_id = _task_key(task, self.client_config)
                append_jsonl(self.raw_path, {
                    "raw_response_id": raw_id,
                    "metric": task.metric,
                    "sample_id": task.sample_id,
                    "task_id": task.task_id,
                    "response": response.to_dict(),
                })
                try:
                    parsed = metric.parse_response(task, response.to_dict())
                except Exception as exc:  # noqa: BLE001 - isolate metric failures
                    parsed = JudgeResult(
                        sample_id=task.sample_id,
                        metric=task.metric,
                        metric_version=task.metric_version,
                        task_id=task.task_id,
                        error={
                            "type": type(exc).__name__,
                            "message": str(exc),
                        },
                    )
                row = parsed.to_dict()
                row["raw_response_id"] = raw_id
                row["params_hash"] = _params_hash(
                    _effective_params(task, self.client_config)
                )
                append_jsonl(self.result_path, row)
                results.append(parsed)
            print(f"[judge] {metric.config.name}: "
                  f"{min(start + batch_size, len(tasks))}/{len(tasks)}")
        return results

    def _load_done_keys(self) -> set[str]:
        done: set[str] = set()
        for row in read_jsonl(self.result_path):
            try:
                done.add(
                    "::".join([
                        row["metric"],
                        row["metric_version"],
                        row["sample_id"],
                        row["task_id"],
                        row.get("params_hash", ""),
                    ])
                )
            except KeyError:
                continue
        return done

    def _load_existing_results(self) -> list[JudgeResult]:
        results: list[JudgeResult] = []
        for row in read_jsonl(self.result_path):
            try:
                results.append(JudgeResult(
                    sample_id=row["sample_id"],
                    metric=row["metric"],
                    metric_version=row["metric_version"],
                    task_id=row["task_id"],
                    score=row.get("score"),
                    passed=row.get("passed"),
                    reason=row.get("reason"),
                    details=row.get("details") or {},
                    error=row.get("error"),
                ))
            except KeyError:
                continue
        return results

    def _build_summary(
        self,
        results: list[JudgeResult],
        total_tasks: int,
        skipped: int,
    ) -> dict[str, Any]:
        by_metric: dict[str, list[JudgeResult]] = defaultdict(list)
        for result in results:
            by_metric[result.metric].append(result)
        metric_summary = {}
        for metric in self.metrics:
            metric_summary[metric.config.name] = metric.aggregate(
                by_metric.get(metric.config.name, [])
            )
        return {
            "created_at": int(time.time()),
            "total_tasks": total_tasks,
            "skipped_tasks": skipped,
            "written_results": len(results),
            "metrics": metric_summary,
        }


def _build_client(config: dict[str, Any]) -> Any:
    backend = str(config.get("backend") or "sglang").lower()
    if backend in {"openai_chat", "api", "chat_completions"}:
        return OpenAIChatClient(
            model=config.get("model"),
            base_url=config.get("base_url"),
            api_key=config.get("api_key"),
            timeout=float(config.get("timeout", 120.0)),
            max_retries=int(config.get("max_retries", 2)),
            retry_delay=float(config.get("retry_delay", 1.0)),
        )
    if backend not in {"sglang", "sglang_generate"}:
        raise ValueError(
            "Unsupported client.backend. Use 'sglang' or 'openai_chat'; "
            f"got {backend!r}"
        )
    return JudgeLLMClient(
        host=config.get("host", "127.0.0.1"),
        port=int(config.get("port", 31877)),
        endpoint=config.get("endpoint", "/generate"),
        base_url=config.get("base_url"),
        api_key=config.get("api_key"),
        model_path=config.get("model_path"),
        tokenize_chat=bool(config.get("tokenize_chat", False)),
        tokenizer_workers=int(config.get("tokenizer_workers", 8)),
        timeout=float(config.get("timeout", 120.0)),
        max_retries=int(config.get("max_retries", 2)),
        retry_delay=float(config.get("retry_delay", 1.0)),
    )


def _resolve_output_dir(output_cfg: dict[str, Any], *, base_dir: Path) -> Path:
    """Resolve output directory.

    Compatibility:
    - output.dir set to a concrete path keeps the old behavior.
    - output.dir omitted/null/"auto" creates a timestamped path under base_dir.
    - output.timestamped=true appends a timestamp under output.dir.

    Relative paths are resolved from the judge package root, not the caller's
    current working directory.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    explicit_dir = output_cfg.get("dir")
    timestamped = bool(output_cfg.get("timestamped", False))

    if explicit_dir and explicit_dir != "auto":
        base = _resolve_path(Path(explicit_dir), base_dir)
        return base / timestamp if timestamped else base

    output_base = _resolve_path(Path(output_cfg.get("base_dir") or "outputs"), base_dir)
    run_name = str(output_cfg.get("run_name") or "run").strip() or "run"
    return output_base / run_name / timestamp


def _resolve_path(path: Path, base_dir: Path) -> Path:
    return path if path.is_absolute() else base_dir / path


def _task_to_payload(task: JudgeTask, client_config: dict[str, Any]) -> dict[str, Any]:
    params = _effective_params(task, client_config)
    use_chat_messages = bool(
        client_config.get("use_chat_messages", client_config.get("tokenize_chat", False))
    )
    if task.messages is not None and use_chat_messages:
        return {"messages": task.messages, **params}
    if task.prompt is not None:
        return {"text": task.prompt, **params}
    raise ValueError(f"Task has neither messages nor prompt: {task}")


def _effective_params(
    task: JudgeTask,
    client_config: dict[str, Any],
) -> dict[str, Any]:
    params = {
        "temperature": client_config.get("temperature", 0.0),
        "max_tokens": client_config.get("max_tokens", 512),
        **task.params,
    }
    for key in ("top_p", "presence_penalty", "frequency_penalty", "stop"):
        if key in client_config and key not in params:
            params[key] = client_config[key]
    if client_config.get("tokenize_chat"):
        chat_template_kwargs = client_config.get("chat_template_kwargs")
        if isinstance(chat_template_kwargs, dict):
            params["chat_template_kwargs"] = chat_template_kwargs
    return params


def _params_hash(params: dict[str, Any]) -> str:
    raw = json.dumps(params, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def _task_key(task: JudgeTask, client_config: dict[str, Any]) -> str:
    return "::".join([
        task.metric,
        task.metric_version,
        task.sample_id,
        task.task_id,
        _params_hash(_effective_params(task, client_config)),
    ])


def _write_empty_summary(output_dir: Path, n_samples: int) -> None:
    summary = {
        "created_at": int(time.time()),
        "n_samples": n_samples,
        "total_tasks": 0,
        "metrics": {},
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

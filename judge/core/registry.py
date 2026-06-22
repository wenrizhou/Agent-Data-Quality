"""Metric plugin discovery and loading."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .io import load_config
from .schemas import JudgeCase, JudgeResult, JudgeTask, MetricConfig


class MetricPlugin(Protocol):
    config: MetricConfig

    def build_tasks(self, sample: JudgeCase) -> list[JudgeTask]:
        ...

    def parse_response(
        self,
        task: JudgeTask,
        response: dict[str, Any],
    ) -> JudgeResult:
        ...

    def aggregate(self, results: list[JudgeResult]) -> dict[str, Any]:
        ...


class BaseMetric:
    """Convenience base class for future metric implementations."""

    def __init__(self, config: MetricConfig) -> None:
        self.config = config

    def build_tasks(self, sample: JudgeCase) -> list[JudgeTask]:
        raise NotImplementedError

    def parse_response(
        self,
        task: JudgeTask,
        response: dict[str, Any],
    ) -> JudgeResult:
        raise NotImplementedError

    def aggregate(self, results: list[JudgeResult]) -> dict[str, Any]:
        scores = [r.score for r in results if r.score is not None]
        return {
            "n_results": len(results),
            "n_scored": len(scores),
            "score_mean": sum(scores) / len(scores) if scores else None,
            "n_errors": sum(1 for r in results if r.error is not None),
        }


@dataclass(frozen=True)
class MetricSpec:
    path: Path
    config_path: Path
    module_path: Path


def discover_metric_dirs(metrics_root: Path) -> dict[str, Path]:
    if not metrics_root.exists():
        return {}
    out: dict[str, Path] = {}
    for path in sorted(metrics_root.iterdir()):
        if path.is_dir() and (path / "metric.yaml").exists():
            out[path.name] = path
    return out


def discover_metric_specs(metrics_root: Path) -> dict[str, MetricSpec]:
    if not metrics_root.exists():
        return {}
    out: dict[str, MetricSpec] = {}
    for path in sorted(metrics_root.iterdir()):
        if not path.is_dir():
            continue

        default_config = path / "metric.yaml"
        default_module = path / "metric.py"
        if default_config.exists() and default_module.exists():
            out[path.name] = MetricSpec(path, default_config, default_module)

        for config_path in sorted(path.glob("*.yaml")):
            if config_path.name == "metric.yaml":
                continue
            raw = load_config(config_path)
            module_name = str(raw.get("module") or f"{config_path.stem}.py")
            module_path = path / module_name
            if not module_path.exists():
                continue
            name = str(raw.get("name") or config_path.stem)
            out.setdefault(name, MetricSpec(path, config_path, module_path))
    return out


def load_metric_config(
    metric_dir: Path,
    overrides: dict[str, Any] | None = None,
    config_path: Path | None = None,
) -> MetricConfig:
    raw = load_config(config_path or metric_dir / "metric.yaml")
    overrides = overrides or {}
    defaults = raw.get("defaults") or {}
    params = {**defaults, **(overrides.get("params") or {})}
    return MetricConfig(
        name=str(raw.get("name") or metric_dir.name),
        version=str(raw.get("version") or "0.1.0"),
        path=str(metric_dir),
        description=raw.get("description"),
        prompt=overrides.get("prompt", raw.get("prompt")),
        output={**(raw.get("output") or {}), **(overrides.get("output") or {})},
        defaults=defaults,
        params=params,
    )


def load_metric(
    metric_dir: Path,
    overrides: dict[str, Any] | None = None,
    config_path: Path | None = None,
    module_path: Path | None = None,
) -> MetricPlugin:
    config = load_metric_config(metric_dir, overrides, config_path)
    metric_py = module_path or metric_dir / "metric.py"
    if not metric_py.exists():
        raise FileNotFoundError(f"Metric missing metric.py: {metric_dir}")

    module_name = f"judge_metric_{metric_dir.name}_{metric_py.stem}"
    spec = importlib.util.spec_from_file_location(module_name, metric_py)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load metric module: {metric_py}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    cls = getattr(module, "Metric", None)
    if cls is None:
        raise AttributeError(f"{metric_py} must define class Metric")
    return cls(config)


def load_configured_metrics(
    metrics_root: Path,
    metric_entries: list[dict[str, Any] | str],
) -> list[MetricPlugin]:
    discovered = discover_metric_specs(metrics_root)
    plugins: list[MetricPlugin] = []
    for entry in metric_entries:
        if isinstance(entry, str):
            entry = {"name": entry}
        name = entry.get("name")
        if not name:
            raise ValueError(f"Metric entry missing name: {entry}")
        metric_dir = discovered.get(name)
        if metric_dir is None:
            raise FileNotFoundError(
                f"Metric {name!r} not found under {metrics_root}"
            )
        plugins.append(
            load_metric(
                metric_dir.path,
                entry,
                config_path=metric_dir.config_path,
                module_path=metric_dir.module_path,
            )
        )
    return plugins

"""Shared dataclasses used by judge metrics and runners."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class JudgeCase:
    """Canonical sample wrapper passed to metrics."""

    sample_id: str
    conversations: list[dict[str, Any]]
    tools: list[dict[str, Any]]
    query: str | None = None
    metadata: dict[str, Any] | None = None
    source_index: int | None = None
    source_path: str | None = None
    source_row: int | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class JudgeTask:
    """One LLM judge request created by a metric."""

    metric: str
    metric_version: str
    sample_id: str
    task_id: str
    messages: list[dict[str, Any]] | None = None
    prompt: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    payload_meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class JudgeResult:
    """Parsed metric result for a single JudgeTask."""

    sample_id: str
    metric: str
    metric_version: str
    task_id: str
    score: float | None = None
    passed: bool | None = None
    reason: str | None = None
    details: dict[str, Any] = field(default_factory=dict)
    error: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MetricConfig:
    """Runtime metric configuration after merging defaults and overrides."""

    name: str
    version: str
    path: str
    description: str | None = None
    prompt: str | None = None
    output: dict[str, Any] = field(default_factory=dict)
    defaults: dict[str, Any] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LLMResponse:
    """Normalized LLM client response."""

    ok: bool
    text: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)
    error: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

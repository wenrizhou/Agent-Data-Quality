"""Prompt template helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def render_template(path: Path, values: dict[str, Any]) -> str:
    """Render a prompt template with Jinja2 if available.

    Falls back to ``str.format`` for simple templates.
    """
    text = path.read_text(encoding="utf-8")
    try:
        from jinja2 import Template
    except ImportError:
        return text.format(**values)
    return Template(text).render(**values)

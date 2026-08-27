"""Loading of the bundled LLMism pattern tables."""

from __future__ import annotations

import re
from dataclasses import dataclass
from importlib import resources
from typing import Any

import yaml

__all__ = ["Pattern", "load_patterns", "compiled_patterns"]

_CATEGORY_LEXICAL = "lexical"
_CATEGORY_PHRASAL = "phrasal"


@dataclass(frozen=True)
class Pattern:
    """One LLMism pattern from ``data/patterns.yaml``."""

    id: str
    category: str
    regex: re.Pattern[str]
    suggestion: str | None


def load_patterns() -> list[Pattern]:
    """Load the bundled seed patterns, compiled and validated."""
    raw = resources.files("llmism.data").joinpath("patterns.yaml").read_text(encoding="utf-8")
    entries: list[dict[str, Any]] = yaml.safe_load(raw) or []
    patterns: list[Pattern] = []
    seen: set[str] = set()
    for entry in entries:
        pid = entry["id"]
        if pid in seen:
            msg = f"duplicate pattern id: {pid}"
            raise ValueError(msg)
        seen.add(pid)
        category = entry["category"]
        if category not in (_CATEGORY_LEXICAL, _CATEGORY_PHRASAL):
            msg = f"pattern {pid}: unknown category {category!r}"
            raise ValueError(msg)
        patterns.append(
            Pattern(
                id=pid,
                category=category,
                regex=re.compile(entry["pattern"], re.IGNORECASE),
                suggestion=entry.get("suggestion"),
            )
        )
    return patterns


def compiled_patterns() -> list[Pattern]:
    """Cached accessor for :func:`load_patterns`."""
    global _CACHE
    if _CACHE is None:
        _CACHE = load_patterns()
    return _CACHE


_CACHE: list[Pattern] | None = None

"""Per-rule configuration loaded from ``.llmism.toml``.

Every linter that survived contact with real teams (ruff, flake8, stylelint)
has ignore/select plus a config file; llmism follows the same shape:

.. code-block:: toml

    # .llmism.toml
    ignore = ["em-dash-overuse", "low-burstiness"]   # disable rules everywhere

    include = [".rst"]                # extra extensions to scan in directory walks
    exclude = ["vendor", "generated"] # extra directory names to skip

    [rules.em-dash-overuse]
    max-per-paragraph = 4                             # tune a threshold

    [rules.leverage-verb]
    suggestion = "rely on"                            # override a replacement

Keys map to the rule's tunables: thresholds for structural rules
(``max-per-paragraph``, ``min-cv``, ...), ``suggestion`` for pattern rules.
Unknown rules or keys are a hard error so typos don't silently no-op.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import tomllib

__all__ = ["Config", "load_config", "ConfigError"]

# Tunables of structural rules, keyed by rule id. Pattern rules (lexical/
# phrasal) have no thresholds; they only accept `suggestion`.
DEFAULTS: dict[str, dict[str, Any]] = {
    "em-dash-overuse": {"max-per-paragraph": 2},
    "bold-lead-in-bullets": {"min-run": 3},
    "bold-sentence-bullet": {},
    "low-burstiness": {"min-cv": 0.25, "min-sentences": 5},
    "uniform-paragraphs": {"min-cv": 0.40, "min-count": 4},
    "mechanical-cadence": {"zigzag": 0.85, "min-cv": 0.25},
    "opener-repetition": {"min": 3},
    "hedge-stacking": {"min": 2},
    "colon-clause": {"label-max-words": 3, "list-min-items": 3},
    "verbless-fragment": {"max-words": 6},
    "sentence-header": {"max-words": 8},
    "synonym-cycling": {"window-words": 300},
    "degenerate-repetition": {"n": 6, "threshold": 0.20},
    "hashtag-stuffing": {"min-tags": 6},
    "emoji": {},
}


class ConfigError(ValueError):
    """Raised when ``.llmism.toml`` references unknown rules or keys."""


@dataclass(frozen=True)
class Config:
    """Effective per-rule configuration after merging ``.llmism.toml``."""

    ignore: frozenset[str] = field(default_factory=frozenset)
    rules: dict[str, dict[str, Any]] = field(default_factory=dict)
    include: frozenset[str] = field(default_factory=frozenset)
    exclude: frozenset[str] = field(default_factory=frozenset)

    def param(self, rule_id: str, key: str) -> Any:
        """Value of ``rules[rule_id][key]``, falling back to the default."""
        if rule_id in DEFAULTS:
            return self.rules.get(rule_id, {}).get(key, DEFAULTS[rule_id][key])
        return self.rules.get(rule_id, {}).get(key)

    def ignored(self, rule_id: str) -> bool:
        return rule_id in self.ignore


def load_config(path: str | Path = ".llmism.toml") -> Config:
    """Load config; a missing or empty file yields the defaults."""
    p = Path(path)
    if not p.is_file():
        return Config()
    data: dict[str, Any] = tomllib.loads(p.read_text(encoding="utf-8"))

    unknown_top = set(data) - {"ignore", "rules", "include", "exclude"}
    if unknown_top:
        msg = f"{p}: unknown top-level key(s) {sorted(unknown_top)}"
        raise ConfigError(msg)

    ignore = frozenset(str(r) for r in data.get("ignore", []))
    if ignore - _VALID_RULE_IDS:
        msg = f"{p}: unknown rule(s) in ignore: {sorted(ignore - _VALID_RULE_IDS)}"
        raise ConfigError(msg)

    include = _ext_list(data.get("include", []), "include", p)
    exclude = _ext_list(data.get("exclude", []), "exclude", p)
    rules: dict[str, dict[str, Any]] = {}
    for rule_id, params in data.get("rules", {}).items():
        if rule_id in DEFAULTS:
            allowed: set[str] = set(DEFAULTS[rule_id])
        elif rule_id in _VALID_RULE_IDS:
            allowed = {"suggestion"}
        else:
            msg = f"{p}: unknown rule {rule_id!r} in [rules]"
            raise ConfigError(msg)
        if not isinstance(params, dict):
            msg = f"{p}: [rules.{rule_id}] must be a table"
            raise ConfigError(msg)
        for key in params:
            if key not in allowed:
                msg = (
                    f"{p}: unknown key {key!r} for rule {rule_id!r} "
                    f"(expected one of {sorted(allowed)})"
                )
                raise ConfigError(msg)
        rules[rule_id] = dict(params)
    return Config(ignore=ignore, rules=rules, include=include, exclude=exclude)


def _ext_list(raw: object, key: str, p: Path) -> frozenset[str]:
    """Validate an include/exclude list of file extensions / dir names."""
    if not isinstance(raw, list) or not all(isinstance(x, str) for x in raw):
        msg = f"{p}: {key} must be a list of strings"
        raise ConfigError(msg)
    return frozenset(raw)


def _pattern_ids() -> frozenset[str]:
    from ._patterns import compiled_patterns

    return frozenset(p.id for p in compiled_patterns())


_VALID_RULE_IDS = frozenset(DEFAULTS) | _pattern_ids()

"""Multi-file support: expand a CLI path into the list of files to scan."""

from __future__ import annotations

from pathlib import Path

from ._config import Config

__all__ = ["expand_targets"]

# extensions llmism knows how to scan; anything else is skipped in tree walks
_SCANNABLE_EXTS = {".md", ".markdown", ".tex", ".txt"}

# never descend into these
_SKIP_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".tox",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "dist",
    "build",
}


def expand_targets(path: str, fmt: str | None = None, config: Config | None = None) -> list[str]:
    """Resolve a CLI ``path`` (file, directory or ``-``) to concrete files.

    Directories are walked recursively for scannable extensions; VCS and
    build dirs are skipped, plus any extra names configured in
    ``exclude``. ``include`` adds extensions to the walk. ``-`` and
    non-existent paths pass through untouched (stdin / explicit user
    intent — the reader raises the error). A forced ``fmt`` narrows the
    walk to matching extensions.
    """
    if path == "-":
        return ["-"]
    p = Path(path)
    if p.is_file() or not p.is_dir():
        return [path]
    skip = _SKIP_DIRS | (config.exclude if config else set())
    exts = _SCANNABLE_EXTS | (set(config.include) if config else set())
    if fmt == "markdown":
        exts = {".md", ".markdown"}
    elif fmt == "latex":
        exts = {".tex"}
    elif fmt == "text":
        exts = {".txt"}
    return sorted(
        str(f)
        for f in p.rglob("*")
        if f.is_file() and f.suffix.lower() in exts and not _in_skip_dir(f, skip)
    )


def _in_skip_dir(f: Path, skip: set[str]) -> bool:
    return any(part in skip for part in f.parts)

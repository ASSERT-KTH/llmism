"""Multi-file support: expand a CLI path into the list of files to scan."""

from __future__ import annotations

from pathlib import Path

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


def expand_targets(path: str, fmt: str | None = None) -> list[str]:
    """Resolve a CLI ``path`` (file, directory or ``-``) to concrete files.

    Directories are walked recursively for scannable extensions; VCS and
    build dirs are skipped. ``-`` and non-existent paths pass through
    untouched (stdin / explicit user intent — the reader raises the error).
    A forced ``fmt`` narrows the walk to matching extensions.
    """
    if path == "-":
        return ["-"]
    p = Path(path)
    if p.is_file() or not p.is_dir():
        return [path]
    exts = _SCANNABLE_EXTS
    if fmt == "markdown":
        exts = {".md", ".markdown"}
    elif fmt == "latex":
        exts = {".tex"}
    elif fmt == "text":
        exts = {".txt"}
    return sorted(
        str(f)
        for f in p.rglob("*")
        if f.is_file() and f.suffix.lower() in exts and not _in_skip_dir(f)
    )


def _in_skip_dir(f: Path) -> bool:
    return any(part in _SKIP_DIRS for part in f.parts)

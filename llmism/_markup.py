"""Format-aware helpers: scannable ranges, bullet/bold lead-ins, sentence splitting."""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = ["Span", "scannable_ranges", "split_sentences", "bold_lead_in_bullets"]

_LATEX_ENV = re.compile(
    r"\\begin\{(verbatim|lstlisting|minted|displaymath|equation\*?|align\*?|"
    r"gather\*?|multline\*?|eqnarray\*?|math|tabular\*?)\}.*?\\end\{\1\}",
    re.DOTALL | re.IGNORECASE,
)
_LATEX_INLINE_MATH = re.compile(r"\$[^$\n]*\$|\\\(.*?\\\)|\\\[.*?\\\]", re.DOTALL)
_MD_FENCE_LINE = re.compile(r"^\s*(```+|~~~+)")
_MD_INLINE_CODE = re.compile(r"`+[^`\n]*`+")

# A bullet line whose first rendered token is a **bold** lead-in.
_BULLET_BOLD_LEAD = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+\*\*[^*\n]+\*\*")

# Sentence boundary: ., ! or ? followed by whitespace, next char not whitespace.
# "3.5" has no space after the dot, so decimals never split.
_SENT_BOUNDARY = re.compile(r"(?<=[.!?])[ \t]+(?=\S)")


@dataclass(frozen=True)
class Span:
    """Half-open char range [start, end) in the original text."""

    start: int
    end: int


def _exclude(text: str, regexes: tuple[re.Pattern[str], ...]) -> list[Span]:
    """Covered ranges of ``text`` after removing every regex match."""
    covered = [(0, len(text))]
    for rx in regexes:
        next_covered: list[tuple[int, int]] = []
        for start, end in covered:
            pos = start
            for m in rx.finditer(text, start, end):
                if m.start() > pos:
                    next_covered.append((pos, m.start()))
                pos = m.end()
            if pos < end:
                next_covered.append((pos, end))
        covered = next_covered
    return [Span(s, e) for s, e in covered if e > s]


def _strip_code_fences(text: str) -> list[Span]:
    """Line-based fence removal: handles unterminated fences too."""
    spans: list[Span] = []
    fence_marker: str | None = None
    run_start = 0
    pos = 0
    for line in text.splitlines(keepends=True):
        m = _MD_FENCE_LINE.match(line)
        if fence_marker is None:
            if m:
                fence_marker = m.group(1)[:3]
                if pos > run_start:
                    spans.append(Span(run_start, pos))
        elif m and m.group(1)[:3] == fence_marker:
            fence_marker = None
            run_start = pos + len(line)
        pos += len(line)
    if fence_marker is None and pos > run_start:
        spans.append(Span(run_start, pos))
    return spans


def scannable_ranges(text: str, fmt: str = "markdown") -> list[Span]:
    """Character ranges of ``text`` that prose rules should scan.

    Code fences and inline code (markdown) and math/verbatim environments
    (LaTeX) are excluded. Plain text scans the whole document.
    """
    if not text:
        return []
    if fmt == "markdown":
        ranges: list[Span] = []
        for span in _strip_code_fences(text):
            sub = text[span.start : span.end]
            ranges.extend(
                Span(span.start + s.start, span.start + s.end)
                for s in _exclude(sub, (_MD_INLINE_CODE,))
            )
        return ranges
    if fmt == "latex":
        return _exclude(text, (_LATEX_ENV, _LATEX_INLINE_MATH))
    return [Span(0, len(text))]


def split_sentences(text: str, fmt: str = "markdown") -> list[tuple[str, int, int]]:
    """Split into ``(sentence, start, end)`` over scannable ranges only."""
    sentences: list[tuple[str, int, int]] = []
    for span in scannable_ranges(text, fmt):
        chunk = text[span.start : span.end]
        seg_start = 0
        for m in _SENT_BOUNDARY.finditer(chunk):
            part = chunk[seg_start : m.start()].strip()
            if part:
                a = chunk.index(part[0], seg_start)
                sentences.append((part, span.start + a, span.start + a + len(part)))
            seg_start = m.end()
        part = chunk[seg_start:].strip()
        if part:
            a = chunk.index(part[0], seg_start)
            sentences.append((part, span.start + a, span.start + a + len(part)))
    return sentences


def bold_lead_in_bullets(text: str, fmt: str = "markdown") -> list[Span]:
    """Runs of >=3 consecutive bullet lines that start with a **bold** lead-in."""
    if fmt != "markdown":
        return []
    line_spans: list[tuple[int, int, bool]] = []
    pos = 0
    for line in text.splitlines(keepends=True):
        line_spans.append((pos, pos + len(line), bool(_BULLET_BOLD_LEAD.match(line))))
        pos += len(line)

    spans: list[Span] = []
    run: list[tuple[int, int, bool]] = []

    def flush() -> None:
        if len(run) >= 3:
            spans.append(Span(run[0][0], run[-1][1]))

    for entry in line_spans:
        if entry[2]:
            run.append(entry)
        else:
            flush()
            run = []
    flush()
    return spans

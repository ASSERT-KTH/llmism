"""Format-aware helpers: scannable ranges, bullets, sentences, paragraphs."""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = [
    "Span",
    "scannable_ranges",
    "split_sentences",
    "split_paragraphs",
    "bold_lead_in_bullets",
]

_LATEX_ENV = re.compile(
    r"\\begin\{(verbatim|lstlisting|minted|displaymath|equation\*?|align\*?|"
    r"gather\*?|multline\*?|eqnarray\*?|math|tabular\*?)\}.*?\\end\{\1\}",
    re.DOTALL | re.IGNORECASE,
)
# '%' to end of line, but not '\%' (escaped percent).
_LATEX_COMMENT = re.compile(r"(?<!\\)%[^\n]*")
_LATEX_INLINE_MATH = re.compile(r"\$[^$\n]*\$|\\\(.*?\\\)|\\\[.*?\\\]", re.DOTALL)
_MD_FENCE_LINE = re.compile(r"^\s*(```+|~~~+)")
_MD_INLINE_CODE = re.compile(r"`+[^`\n]*`+")
_MD_HEADING = re.compile(r"^#{1,6}\s.*$", re.MULTILINE)

# A bullet line whose first rendered token is a **bold** lead-in.
_BULLET_BOLD_LEAD = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+\*\*[^*\n]+\*\*")

# Sentence boundary: ., ! or ? followed by whitespace, next char not whitespace.
# "3.5" has no space after the dot, so decimals never split.
_SENT_BOUNDARY = re.compile(r"(?<=[.!?])[ \t]+(?=\S)")

_ABBREV = re.compile(r"\b(?:e\.g|i\.e|etc|vs|cf|Dr|Prof|Mr|Mrs|Ms|Fig|No|pp|Vol)\.$")


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

    Code fences and inline code (markdown) and math environments, verbatim
    blocks and ``%`` comments (LaTeX) are excluded. Plain text scans the
    whole document.
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
        return _exclude(text, (_LATEX_ENV, _LATEX_COMMENT, _LATEX_INLINE_MATH))
    return [Span(0, len(text))]


def split_sentences(text: str, fmt: str = "markdown") -> list[tuple[str, int, int]]:
    """Split into ``(sentence, start, end)`` over scannable ranges only."""
    sentences: list[tuple[str, int, int]] = []
    for span in scannable_ranges(text, fmt):
        chunk = text[span.start : span.end]
        seg_start = 0
        for m in _SENT_BOUNDARY.finditer(chunk):
            part = chunk[seg_start : m.start()].strip()
            a = chunk.index(part[0], seg_start) if part else m.start()
            if part and not _ABBREV.search(part):
                sentences.append((part, span.start + a, span.start + a + len(part)))
            elif part:
                # abbreviation ended the fragment: glue it to the next sentence
                seg_end = m.end()
                nxt = _SENT_BOUNDARY.search(chunk, seg_end)
                end = nxt.start() if nxt else len(chunk)
                glued = chunk[seg_start:end].strip()
                if glued:
                    ga = chunk.index(glued[0], seg_start)
                    sentences.append((glued, span.start + ga, span.start + ga + len(glued)))
                seg_start = end
                continue
            seg_start = m.end()
        part = chunk[seg_start:].strip()
        if part:
            a = chunk.index(part[0], seg_start)
            sentences.append((part, span.start + a, span.start + a + len(part)))
    return sentences


def split_paragraphs(text: str, fmt: str = "markdown") -> list[tuple[str, int, int]]:
    """Split into ``(paragraph, start, end)`` over scannable ranges only.

    For markdown, ATX headings start a new paragraph and standalone
    horizontal rules are ignored. For LaTeX, ``\\\\`` does not split.
    """
    paragraphs: list[tuple[str, int, int]] = []
    for span in scannable_ranges(text, fmt):
        chunk = text[span.start : span.end]
        pos = span.start
        current_start: int | None = None
        current: list[str] = []

        def flush(cstart: int | None, clines: list[str]) -> None:
            if cstart is not None:
                joined = "".join(clines)
                stripped = joined.strip()
                if stripped:
                    offset = cstart + (len(joined) - len(joined.lstrip()))
                    end = cstart + len(joined.rstrip())
                    paragraphs.append((stripped, offset, end))

        for line in chunk.splitlines(keepends=True):
            is_break = (
                fmt == "markdown"
                and (line.startswith("#") or re.match(r"^\s*(?:-{3,}|\*{3,}|_{3,})\s*$", line))
            ) or line.strip() == ""
            if is_break:
                flush(current_start, current)
                current_start = None
                current = []
            else:
                if current_start is None:
                    current_start = pos
                current.append(line)
            pos += len(line)
        flush(current_start, current)
    return paragraphs


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

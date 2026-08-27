"""LLMism detector: lexical, phrasal and structural findings in prose."""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass

from ._markup import Span, bold_lead_in_bullets, scannable_ranges, split_sentences
from ._patterns import Pattern, compiled_patterns

__all__ = ["Finding", "Detector"]

CATEGORIES = ("lexical", "phrasal", "structural")

# --- structural thresholds (tunable) -----------------------------------------
EM_DASH_MAX_PER_100_WORDS = 1.0
MIN_WORDS_FOR_EM_DASH = 40
EM_DASH = "\N{EM DASH}"
EN_DASH_PAIR = re.compile(r"\s--\s")  # double hyphen is an em-dash surrogate
BOLD_LEAD_IN_MIN_RUN = 3
BURSTINESS_MIN_CV = 0.25  # stdev/mean of sentence length (in words)
BURSTINESS_MIN_SENTENCES = 5
_RQ_ANSWER_START = re.compile(
    r"^\s*(?:the answer is|yes[,.]|no[,.]|it is|that'?s because)\b", re.IGNORECASE
)
_SENT_END_Q = re.compile(r"\?\s*$")
_ELLIPSIS_ANSWER = re.compile(r"^\s*(?:the answer is|yes|no)\b", re.IGNORECASE)


@dataclass(frozen=True)
class Finding:
    """One detected LLMism.

    ``suggestion`` is a safe rule-based replacement; ``None`` means no
    deterministic rewrite exists (fix by hand or with ``--llm``).
    """

    category: str
    rule_id: str
    start: int
    end: int
    matched_text: str
    message: str
    suggestion: str | None = None


class Detector:
    """Scans text for LLM-sounding lexical, phrasal and structural tics."""

    def __init__(self, patterns: list[Pattern] | None = None) -> None:
        self._patterns = patterns if patterns is not None else compiled_patterns()

    def scan(self, text: str, fmt: str = "markdown") -> list[Finding]:
        ranges = scannable_ranges(text, fmt)
        findings: list[Finding] = []
        findings.extend(self._scan_patterns(text, ranges))
        findings.extend(self._scan_em_dash(text, ranges))
        findings.extend(self._scan_bold_lead_ins(text, fmt))
        findings.extend(self._scan_rhetorical_questions(text, fmt))
        findings.extend(self._scan_burstiness(text, fmt))
        return sorted(findings, key=lambda f: (f.start, f.end))

    # -- lexical + phrasal ---------------------------------------------------
    def _scan_patterns(self, text: str, ranges: list[Span]) -> list[Finding]:
        findings: list[Finding] = []
        for pat in self._patterns:
            for m in pat.regex.finditer(text):
                if not any(s.start <= m.start() and m.end() <= s.end for s in ranges):
                    continue
                findings.append(
                    Finding(
                        category=pat.category,
                        rule_id=pat.id,
                        start=m.start(),
                        end=m.end(),
                        matched_text=m.group(0),
                        message=self._message(pat),
                        suggestion=pat.suggestion,
                    )
                )
        return findings

    @staticmethod
    def _message(pat: Pattern) -> str:
        kind = "Overused word" if pat.category == "lexical" else "Overused phrase"
        if pat.suggestion:
            return f"{kind}: consider '{pat.suggestion}'"
        return f"{kind}: LLM-sounding pattern, rewrite manually"

    # -- structural -----------------------------------------------------------
    def _scan_em_dash(self, text: str, ranges: list[Span]) -> list[Finding]:
        findings: list[Finding] = []
        for span in ranges:
            chunk = text[span.start : span.end]
            words = len(chunk.split())
            if words < MIN_WORDS_FOR_EM_DASH:
                continue
            hits = [m.start() for m in re.finditer(re.escape(EM_DASH), chunk)]
            hits += [m.start() for m in EN_DASH_PAIR.finditer(chunk)]
            if len(hits) > max(1, int(words * EM_DASH_MAX_PER_100_WORDS / 100)):
                first = span.start + hits[0]
                last = span.start + hits[-1]
                findings.append(
                    Finding(
                        category="structural",
                        rule_id="em-dash-density",
                        start=first,
                        end=last,
                        matched_text=text[first:last],
                        message=(
                            f"{len(hits)} em-dashes in {words} words "
                            f"(threshold {EM_DASH_MAX_PER_100_WORDS}/100 words)"
                        ),
                    )
                )
        return findings

    def _scan_bold_lead_ins(self, text: str, fmt: str) -> list[Finding]:
        return [
            Finding(
                category="structural",
                rule_id="bold-lead-in-bullets",
                start=span.start,
                end=span.end,
                matched_text=text[span.start : span.end],
                message=f"{BOLD_LEAD_IN_MIN_RUN}+ consecutive bullets start with a bold lead-in",
            )
            for span in bold_lead_in_bullets(text, fmt)
        ]

    def _scan_rhetorical_questions(self, text: str, fmt: str) -> list[Finding]:
        sentences = split_sentences(text, fmt)
        findings: list[Finding] = []
        for (q, qs, _qe), (a, _as, ae) in zip(sentences, sentences[1:], strict=False):
            if not _SENT_END_Q.search(q):
                continue
            if not (_RQ_ANSWER_START.match(a) or _ELLIPSIS_ANSWER.match(a)):
                continue
            findings.append(
                Finding(
                    category="structural",
                    rule_id="rhetorical-question-answer",
                    start=qs,
                    end=ae,
                    matched_text=text[qs:ae],
                    message="Rhetorical question immediately answered; restate directly",
                )
            )
        return findings

    def _scan_burstiness(self, text: str, fmt: str) -> list[Finding]:
        sentences = split_sentences(text, fmt)
        lengths = [len(s.split()) for s, _start, _end in sentences]
        if len(lengths) < BURSTINESS_MIN_SENTENCES:
            return []
        mean = statistics.mean(lengths)
        if mean == 0:
            return []
        cv = statistics.pstdev(lengths) / mean
        if cv < BURSTINESS_MIN_CV:
            return [
                Finding(
                    category="structural",
                    rule_id="low-burstiness",
                    start=0,
                    end=len(text),
                    matched_text="",
                    message=(
                        f"Uniform sentence rhythm (length CV {cv:.2f} < "
                        f"{BURSTINESS_MIN_CV:.2f}); vary sentence lengths"
                    ),
                )
            ]
        return []

    # -- helpers ---------------------------------------------------------------
    @staticmethod
    def pattern_ids() -> list[str]:
        return [p.id for p in compiled_patterns()]

"""LLMism detector: lexical, phrasal and structural findings in prose."""

from __future__ import annotations

import collections
import re
import statistics
from dataclasses import dataclass

from ._markup import (
    Span,
    bold_lead_in_bullets,
    scannable_ranges,
    split_paragraphs,
    split_sentences,
)
from ._patterns import Pattern, compiled_patterns

__all__ = ["Finding", "Detector"]

CATEGORIES = ("lexical", "phrasal", "structural")

# --- structural thresholds (tunable) -----------------------------------------
EM_DASH_MAX_PER_PARAGRAPH = 2
BOLD_LEAD_IN_MIN_RUN = 3
BURSTINESS_MIN_CV = 0.25  # stdev/mean of sentence length (in words)
BURSTINESS_MIN_SENTENCES = 5
PARAGRAPH_MIN_CV = 0.40
PARAGRAPH_MIN_COUNT = 4
CADENCE_ZIGZAG = 0.85  # share of successive length-difference sign flips
CADENCE_MIN_CV = 0.25
OPENER_REPETITION_MIN = 3
TRANSITION_CLUSTER_MIN = 2
REP_N = 6  # n-gram size for degenerate repetition
REP_MIN_WORDS = REP_N * 3
REP_THRESHOLD = 0.20
HEDGE_STACK_MIN = 2
MIN_WORDS_FOR_SYN_CLUSTERS = 40

EM_DASH = "\N{EM DASH}"

_TRANSITION_OPENER = re.compile(
    r"(?:^|\n|[.!?]\s+)(Additionally|Furthermore|Moreover|However|Nevertheless|"
    r"Consequently|Subsequently|Notably)\b"
)
_HEDGE = re.compile(
    r"\b(?:may|might|could|possibly|potentially|arguably|perhaps|"
    r"seems?(?:\s+(?:to|like))?|appears?\s+to|somewhat|relatively|fairly|"
    r"kind of|sort of)\b",
    re.IGNORECASE,
)
_RQ_ANSWER_START = re.compile(
    r"^\s*(?:the answer is|yes[,.]|no[,.]|it is|that'?s because)\b", re.IGNORECASE
)
_SENT_END_Q = re.compile(r"\?\s*$")
_REP_WORD = re.compile(r"[A-Za-z][A-Za-z'-]*")
_SYNONYM_CLUSTERS = [
    ("protagonist", "main character", "central figure", "hero", "heroine"),
    ("company", "firm", "organization", "enterprise", "business"),
    ("author", "writer", "novelist", "scribe"),
    ("report", "study", "analysis", "investigation", "examination"),
    ("technology", "tool", "system", "platform", "solution"),
    ("technique", "method", "approach", "strategy", "tactic"),
    ("challenge", "obstacle", "difficulty", "hurdle", "barrier"),
    ("benefit", "advantage", "upside", "strength"),
]


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
        findings.extend(self._scan_bold_lead_ins(text, fmt))
        findings.extend(self._scan_paragraph_rules(text, fmt))
        findings.extend(self._scan_rhetorical_questions(text, fmt))
        findings.extend(self._scan_burstiness(text, fmt))
        findings.extend(self._scan_paragraph_monotony(text, fmt))
        findings.extend(self._scan_cadence(text, fmt))
        findings.extend(self._scan_synonym_cycling(text, fmt))
        findings.extend(self._scan_degenerate_repetition(text, fmt))
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

    # -- structural: bullets ---------------------------------------------------
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

    # -- structural: per-paragraph (em-dash, transitions, openers, hedges) -----
    def _scan_paragraph_rules(self, text: str, fmt: str) -> list[Finding]:
        findings: list[Finding] = []
        for para, p_start, p_end in split_paragraphs(text, fmt):
            # em-dash overuse
            count = para.count(EM_DASH)
            if count > EM_DASH_MAX_PER_PARAGRAPH:
                first = para.find(EM_DASH)
                last = para.rfind(EM_DASH) + len(EM_DASH)
                findings.append(
                    Finding(
                        category="structural",
                        rule_id="em-dash-overuse",
                        start=p_start + first,
                        end=p_start + last,
                        matched_text=text[p_start + first : p_start + last],
                        message=(
                            f"{count} em-dashes in one paragraph "
                            f"(threshold {EM_DASH_MAX_PER_PARAGRAPH})"
                        ),
                    )
                )
            # transition-word cluster
            transitions = list(_TRANSITION_OPENER.finditer(para))
            if len(transitions) >= TRANSITION_CLUSTER_MIN:
                findings.append(
                    Finding(
                        category="structural",
                        rule_id="transition-cluster",
                        start=p_start + transitions[0].start(1),
                        end=p_start + transitions[-1].end(1),
                        matched_text=", ".join(m.group(1) for m in transitions),
                        message=(
                            f"{len(transitions)} stacked transition words in one paragraph "
                            f"({', '.join(m.group(1) for m in transitions)})"
                        ),
                    )
                )
            # sentence-opener repetition
            sents = [s for s, _a, _b in split_sentences(para, "text")]
            if len(sents) >= OPENER_REPETITION_MIN:
                opener_counts: dict[str, int] = {}
                for sent in sents:
                    words = sent.split()
                    if len(words) >= 2:
                        opener = " ".join(words[:2])
                        opener_counts[opener.lower()] = opener_counts.get(opener.lower(), 0) + 1
                for opener, n in opener_counts.items():
                    if n >= OPENER_REPETITION_MIN:
                        findings.append(
                            Finding(
                                category="structural",
                                rule_id="opener-repetition",
                                start=p_start,
                                end=p_end,
                                matched_text=f"{opener!r} x{n}",
                                message=f"{n} of {len(sents)} sentences open with {opener!r}",
                            )
                        )
            # hedge stacking
            for sent, s_start, s_end in split_sentences(para, "text"):
                hedges = _HEDGE.findall(sent)
                if len(hedges) >= HEDGE_STACK_MIN:
                    first_h = _HEDGE.search(sent)
                    findings.append(
                        Finding(
                            category="structural",
                            rule_id="hedge-stacking",
                            start=p_start + s_start + (first_h.start() if first_h else 0),
                            end=p_start + s_end,
                            matched_text=" + ".join(hedges),
                            message=(
                                f"{len(hedges)} stacked hedges in one sentence "
                                f"({', '.join(hedges)})"
                            ),
                        )
                    )
        return findings

    # -- structural: rhetorical Q&A -------------------------------------------
    def _scan_rhetorical_questions(self, text: str, fmt: str) -> list[Finding]:
        sentences = split_sentences(text, fmt)
        findings: list[Finding] = []
        for (q, qs, _qe), (a, _as, ae) in zip(sentences, sentences[1:], strict=False):
            if not _SENT_END_Q.search(q):
                continue
            if not _RQ_ANSWER_START.match(a):
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

    # -- structural: document-level rhythm --------------------------------------
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

    def _scan_paragraph_monotony(self, text: str, fmt: str) -> list[Finding]:
        paragraphs = split_paragraphs(text, fmt)
        lengths = [len(p.split()) for p, _a, _b in paragraphs if p.split()]
        if len(lengths) < PARAGRAPH_MIN_COUNT:
            return []
        mean = statistics.mean(lengths)
        if mean == 0:
            return []
        cv = statistics.pstdev(lengths) / mean
        if cv < PARAGRAPH_MIN_CV:
            return [
                Finding(
                    category="structural",
                    rule_id="uniform-paragraphs",
                    start=0,
                    end=len(text),
                    matched_text="",
                    message=(
                        f"Uniform paragraph sizes (CV {cv:.2f} < "
                        f"{PARAGRAPH_MIN_CV:.2f} over {len(lengths)} paragraphs)"
                    ),
                )
            ]
        return []

    def _scan_cadence(self, text: str, fmt: str) -> list[Finding]:
        """Mechanical short-long-short-long alternation that defeats CV checks."""
        lengths = [len(s.split()) for s, _a, _b in split_sentences(text, fmt) if s.split()]
        if len(lengths) < 6:
            return []
        signs = [
            1 if b > a else -1 if b < a else 0 for a, b in zip(lengths, lengths[1:], strict=False)
        ]
        pairs = list(zip(signs, signs[1:], strict=False))
        if not pairs:
            return []
        flips = sum(1 for a, b in pairs if a and b and a == -b)
        zigzag = flips / len(pairs)
        mean = statistics.mean(lengths)
        cv = statistics.pstdev(lengths) / mean if mean else 0.0
        if zigzag >= CADENCE_ZIGZAG and cv >= CADENCE_MIN_CV:
            return [
                Finding(
                    category="structural",
                    rule_id="mechanical-cadence",
                    start=0,
                    end=len(text),
                    matched_text="",
                    message=(
                        f"Periodic short/long sentence alternation (zigzag {zigzag:.2f}, "
                        f"CV {cv:.2f}); the rhythm is metronomic"
                    ),
                )
            ]
        return []

    # -- structural: vocabulary-level ------------------------------------------
    def _scan_synonym_cycling(self, text: str, fmt: str) -> list[Finding]:
        """Three or more synonyms from one cluster used for the same referent."""
        chunks = [text[s.start : s.end] for s in scannable_ranges(text, fmt)]
        lower = "\n".join(chunks).lower()
        if len(lower.split()) < MIN_WORDS_FOR_SYN_CLUSTERS:
            return []
        findings: list[Finding] = []
        for cluster in _SYNONYM_CLUSTERS:
            used = [t for t in cluster if re.search(rf"\b{re.escape(t)}s?\b", lower)]
            if len(used) >= 3:
                findings.append(
                    Finding(
                        category="structural",
                        rule_id="synonym-cycling",
                        start=0,
                        end=len(text),
                        matched_text=", ".join(used),
                        message=(
                            "Synonym cycling: "
                            f"{len(used)} of {', '.join(used)} used interchangeably"
                        ),
                    )
                )
        return findings

    def _scan_degenerate_repetition(self, text: str, fmt: str) -> list[Finding]:
        """Share of six-word spans occurring more than once (model loop)."""
        words = [
            w.lower()
            for s in scannable_ranges(text, fmt)
            for w in _REP_WORD.findall(text[s.start : s.end])
        ]
        if len(words) < REP_MIN_WORDS:
            return []
        grams = [tuple(words[i : i + REP_N]) for i in range(len(words) - REP_N + 1)]
        counts = collections.Counter(grams)
        dup = sum(c for g, c in counts.items() if c > 1)
        ratio = dup / len(grams)
        if ratio > REP_THRESHOLD:
            return [
                Finding(
                    category="structural",
                    rule_id="degenerate-repetition",
                    start=0,
                    end=len(text),
                    matched_text="",
                    message=(
                        f"{ratio:.0%} of {REP_N}-word spans repeat (threshold {REP_THRESHOLD:.0%})"
                    ),
                )
            ]
        return []

    # -- helpers ---------------------------------------------------------------
    @staticmethod
    def pattern_ids() -> list[str]:
        return [p.id for p in compiled_patterns()]

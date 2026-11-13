"""LLMism detector: lexical, phrasal and structural findings in prose."""

from __future__ import annotations

import collections
import re
import statistics
from dataclasses import dataclass

from ._config import DEFAULTS, Config
from ._markup import (
    Span,
    bold_lead_in_bullets,
    bold_sentence_bullets,
    scannable_ranges,
    split_paragraphs,
    split_sentences,
)
from ._patterns import Pattern, compiled_patterns

__all__ = ["Finding", "Detector"]

CATEGORIES = ("lexical", "phrasal", "structural")

# --- structural thresholds: defaults live in _config.DEFAULTS, override in .llmism.toml
EM_DASH_MAX_PER_PARAGRAPH = DEFAULTS["em-dash-overuse"]["max-per-paragraph"]
BOLD_LEAD_IN_MIN_RUN = DEFAULTS["bold-lead-in-bullets"]["min-run"]
BURSTINESS_MIN_CV = DEFAULTS["low-burstiness"]["min-cv"]
BURSTINESS_MIN_SENTENCES = DEFAULTS["low-burstiness"]["min-sentences"]
PARAGRAPH_MIN_CV = DEFAULTS["uniform-paragraphs"]["min-cv"]
PARAGRAPH_MIN_COUNT = DEFAULTS["uniform-paragraphs"]["min-count"]
CADENCE_ZIGZAG = DEFAULTS["mechanical-cadence"]["zigzag"]
CADENCE_MIN_CV = DEFAULTS["mechanical-cadence"]["min-cv"]
OPENER_REPETITION_MIN = DEFAULTS["opener-repetition"]["min"]
REP_N = DEFAULTS["degenerate-repetition"]["n"]
REP_MIN_WORDS = REP_N * 3
REP_THRESHOLD = DEFAULTS["degenerate-repetition"]["threshold"]
HEDGE_STACK_MIN = DEFAULTS["hedge-stacking"]["min"]
MIN_WORDS_FOR_SYN_CLUSTERS = 40
SYNONYM_WINDOW_WORDS = DEFAULTS["synonym-cycling"]["window-words"]
COLON_LIST_MIN_ITEMS = DEFAULTS["colon-clause"]["list-min-items"]
COLON_LABEL_MAX_WORDS = DEFAULTS["colon-clause"]["label-max-words"]
FRAGMENT_MAX_WORDS = DEFAULTS["verbless-fragment"]["max-words"]
HEADER_MAX_WORDS = DEFAULTS["sentence-header"]["max-words"]

EM_DASH = "\N{EM DASH}"

_HEDGE = re.compile(
    r"\b(?:may|might|could|possibly|potentially|arguably|perhaps|"
    r"seems?(?:\s+(?:to|like))?|appears?\s+to|somewhat|relatively|fairly|"
    r"kind of|sort of)\b",
    re.IGNORECASE,
)
# colon-hinged sentences: a colon is only legitimate before a list of 3+ items
_COLON = re.compile(r":")
_LIST_MARKER = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s*")
_SKIP_LINE = re.compile(r"^\s*(?:#{1,6}\s|\||>\s|```)")
_URL_COLON = re.compile(r"://|\bhttps?$|\bftp$|\bmailto$")
_TIME_COLON = re.compile(r"\d$")
# code, shell and hex-dump lines are not prose, whatever punctuation they carry
_CODEY = re.compile(r"[=<>{}\[\]|$\\]|0x|::|\w\(\)|/\w+/")

# verbless fragments used as sentences ("Two things worth watching.")
_FRAGMENT_OPENER = re.compile(
    r"^(?:the|a|an|one|two|three|four|another|no|same|other|both|this|that|these|those)\b",
    re.IGNORECASE,
)
_FINITE_VERB = re.compile(
    r"\b(?:is|are|was|were|be|been|being|am|has|have|had|do|does|did|can|could|"
    r"will|would|shall|should|may|might|must|go|goes|get|gets|comes?|makes?|"
    r"means?|needs?|says?|shows?|takes?|works?|\w+ed|"
    r"found|made|said|told|took|went|came|saw|got|gave|ran|won|held|left|"
    r"built|sent|kept|meant|felt|put|set|cut|read|led|shown|done|seen|known)\b",
    re.IGNORECASE,
)

# headers are labels, not sentences
_MD_HEADER = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)
_REP_WORD = re.compile(r"[A-Za-z][A-Za-z'-]*")
# Pictographic emoji, plus the joiners/selectors/skin tones that glue a run
# together. Text symbols (© ® ™ → × ≥) are not emoji and stay out.
_EMOJI_BASE = (
    "\U0001f000-\U0001faff\u2600-\u27bf\u231a\u231b\u23e9-\u23fa"
    "\u2b05-\u2b07\u2b1b\u2b1c\u2b50\u2b55"
)
_EMOJI = re.compile(f"[{_EMOJI_BASE}](?:[{_EMOJI_BASE}\u200d\ufe0f]+)?")
# A trailing run of social tags, not ordinary Markdown headings or inline tags.
HASHTAG_MIN_TAGS = DEFAULTS["hashtag-stuffing"]["min-tags"]
# config key -> module constant, for _apply_to_thresholds
_PARAM_TO_CONST = {
    ("em-dash-overuse", "max-per-paragraph"): "EM_DASH_MAX_PER_PARAGRAPH",
    ("bold-lead-in-bullets", "min-run"): "BOLD_LEAD_IN_MIN_RUN",
    ("low-burstiness", "min-cv"): "BURSTINESS_MIN_CV",
    ("low-burstiness", "min-sentences"): "BURSTINESS_MIN_SENTENCES",
    ("uniform-paragraphs", "min-cv"): "PARAGRAPH_MIN_CV",
    ("uniform-paragraphs", "min-count"): "PARAGRAPH_MIN_COUNT",
    ("mechanical-cadence", "zigzag"): "CADENCE_ZIGZAG",
    ("mechanical-cadence", "min-cv"): "CADENCE_MIN_CV",
    ("opener-repetition", "min"): "OPENER_REPETITION_MIN",
    ("hedge-stacking", "min"): "HEDGE_STACK_MIN",
    ("colon-clause", "label-max-words"): "COLON_LABEL_MAX_WORDS",
    ("colon-clause", "list-min-items"): "COLON_LIST_MIN_ITEMS",
    ("verbless-fragment", "max-words"): "FRAGMENT_MAX_WORDS",
    ("sentence-header", "max-words"): "HEADER_MAX_WORDS",
    ("synonym-cycling", "window-words"): "SYNONYM_WINDOW_WORDS",
    ("degenerate-repetition", "n"): "REP_N",
    ("degenerate-repetition", "threshold"): "REP_THRESHOLD",
    ("hashtag-stuffing", "min-tags"): "HASHTAG_MIN_TAGS",
}
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
    """Scans text for LLM-sounding lexical, phrasal and structural tics.

    ``config`` carries per-rule settings from ``.llmism.toml`` (see
    :mod:`llmism._config`); ``None`` uses all defaults with nothing ignored.
    """

    def __init__(
        self, patterns: list[Pattern] | None = None, config: Config | None = None
    ) -> None:
        self.config = config or Config()
        if patterns is None:
            patterns = self._apply_to_patterns(compiled_patterns())
        self._patterns = patterns
        self._apply_to_thresholds()

    def _apply_to_patterns(self, patterns: list[Pattern]) -> list[Pattern]:
        """Drop ignored rules; apply ``suggestion`` overrides from config."""
        from dataclasses import replace

        out: list[Pattern] = []
        for p in patterns:
            if self.config.ignored(p.id):
                continue
            override = self.config.rules.get(p.id, {}).get("suggestion")
            if override is not None and override != p.suggestion:
                out.append(replace(p, suggestion=override))
            else:
                out.append(p)
        return out

    def _apply_to_thresholds(self) -> None:
        """Bind defaults, then config overrides, onto module constants."""
        import llmism.detector as d

        g = d.__dict__
        for (rule_id, key), const in _PARAM_TO_CONST.items():
            g[const] = self.config.param(rule_id, key)
        g["REP_MIN_WORDS"] = g["REP_N"] * 3

    def scan(self, text: str, fmt: str = "markdown") -> list[Finding]:
        ranges = scannable_ranges(text, fmt)
        findings: list[Finding] = []
        findings.extend(self._scan_patterns(text, ranges))
        findings.extend(self._scan_bold_lead_ins(text, fmt))
        findings.extend(self._scan_bold_sentence_bullets(text, fmt))
        findings.extend(self._scan_emoji(text, ranges))
        findings.extend(self._scan_paragraph_rules(text, fmt))
        findings.extend(self._scan_colon_clauses(text, ranges))
        findings.extend(self._scan_verbless_fragments(text, fmt))
        findings.extend(self._scan_sentence_headers(text, fmt, ranges))
        findings.extend(self._scan_burstiness(text, fmt))
        findings.extend(self._scan_paragraph_monotony(text, fmt))
        findings.extend(self._scan_cadence(text, fmt))
        findings.extend(self._scan_synonym_cycling(text, fmt))
        findings.extend(self._scan_degenerate_repetition(text, fmt))
        findings.extend(self._scan_hashtag_stuffing(text, ranges))
        if self.config.ignore:
            findings = [f for f in findings if not self.config.ignored(f.rule_id)]
        return sorted(findings, key=lambda f: (f.start, f.end))

    def _scan_hashtag_stuffing(self, text: str, ranges: list[Span]) -> list[Finding]:
        """Flag lines made of many social tags outside protected markup."""
        findings: list[Finding] = []
        run_re = re.compile(
            rf"(?:^|\n)([ \t]*(?:#[A-Za-z][\w-]*[ \t]+){{{HASHTAG_MIN_TAGS - 1},}}"
            rf"#[A-Za-z][\w-]*[ \t]*)(?=\n|\Z)"
        )
        for match in run_re.finditer(text):
            start, end = match.span(1)
            if not any(span.start <= start and end <= span.end for span in ranges):
                continue
            findings.append(
                Finding(
                    category="structural",
                    rule_id="hashtag-stuffing",
                    start=start,
                    end=end,
                    matched_text=text[start:end],
                    message=f"{HASHTAG_MIN_TAGS} or more tags in one line; keep only useful tags",
                )
            )
        return findings

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

    def _scan_bold_sentence_bullets(self, text: str, fmt: str) -> list[Finding]:
        return [
            Finding(
                category="structural",
                rule_id="bold-sentence-bullet",
                start=span.start,
                end=span.end,
                matched_text=text[span.start : span.end],
                message=(
                    "Bullet opens with a bolded sentence; drop the bold or write "
                    "the item as plain prose"
                ),
            )
            for span in bold_sentence_bullets(text, fmt)
        ]

    def _scan_emoji(self, text: str, ranges: list[Span]) -> list[Finding]:
        """Each run of emoji outside code; the fix deletes it."""
        return [
            Finding(
                category="structural",
                rule_id="emoji",
                start=m.start(),
                end=m.end(),
                matched_text=m.group(0),
                message="Emoji in prose; delete it",
                suggestion="",
            )
            for m in _EMOJI.finditer(text)
            if any(s.start <= m.start() and m.end() <= s.end for s in ranges)
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

    # -- structural: colon-hinged sentences -------------------------------------
    def _scan_colon_clauses(self, text: str, ranges: list[Span]) -> list[Finding]:
        """A colon joining a label to a clause, outside a list of 3+ items."""
        findings: list[Finding] = []
        for m in _COLON.finditer(text):
            pos = m.start()
            if not any(s.start <= pos < s.end for s in ranges):
                continue
            line_start = text.rfind("\n", 0, pos) + 1
            line_end = text.find("\n", pos)
            line_end = len(text) if line_end == -1 else line_end
            line = text[line_start:line_end]
            if _SKIP_LINE.match(line):
                continue
            left = text[line_start:pos]
            right = text[pos + 1 : line_end]
            if _URL_COLON.search(left) or (_TIME_COLON.search(left) and right[:1].isdigit()):
                continue
            if _CODEY.search(line):
                continue  # shell, hex dumps, expressions: not prose
            if not right.strip():  # label line, payload on the next line
                continue
            label = _LIST_MARKER.sub("", left)
            if len(label.split()) <= COLON_LABEL_MAX_WORDS:
                continue  # "Note: ..." style key-value spec, not a hinged sentence
            clause = right.split(".")[0]
            if clause.count(",") >= COLON_LIST_MIN_ITEMS - 1 or clause.count(";") >= 2:
                continue  # literal list of three or more items
            if len(clause.split()) < 3:
                continue
            findings.append(
                Finding(
                    category="structural",
                    rule_id="colon-clause",
                    start=pos,
                    end=min(pos + 1 + len(right), line_end),
                    matched_text=text[max(line_start, pos - 30) : min(line_end, pos + 30)],
                    message=(
                        "Colon hinges a label onto a clause; split into two sentences "
                        "or join with because/so/but/and"
                    ),
                )
            )
        return findings

    # -- structural: verbless fragments -----------------------------------------
    def _scan_verbless_fragments(self, text: str, fmt: str) -> list[Finding]:
        """Noun-phrase fragments standing in for sentences."""
        findings: list[Finding] = []
        for sent, start, end in split_sentences(text, fmt):
            stripped = sent.strip()
            if not stripped.endswith("."):
                continue
            words = stripped.rstrip(".").split()
            if not 2 <= len(words) <= FRAGMENT_MAX_WORDS:
                continue
            if not _FRAGMENT_OPENER.match(stripped):
                continue
            if _FINITE_VERB.search(stripped) or _CODEY.search(stripped) or "`" in stripped:
                continue
            findings.append(
                Finding(
                    category="structural",
                    rule_id="verbless-fragment",
                    start=start,
                    end=end,
                    matched_text=stripped,
                    message=(
                        "Verbless fragment used as a sentence; fold it into the "
                        "sentence it introduces"
                    ),
                )
            )
        return findings

    # -- structural: headers ----------------------------------------------------
    def _scan_sentence_headers(self, text: str, fmt: str, ranges: list[Span]) -> list[Finding]:
        """Markdown headers written as sentences rather than labels."""
        if fmt != "markdown":
            return []
        findings: list[Finding] = []
        for m in _MD_HEADER.finditer(text):
            if not any(s.start <= m.start() and m.end() <= s.end for s in ranges):
                continue  # shell comment inside a code fence, not a header
            title = m.group(2)
            words = title.split()
            too_long = len(words) > HEADER_MAX_WORDS
            punctuated = title.endswith((".", "!", "?"))
            if not (too_long or punctuated):
                continue
            reason = f"{len(words)} words" if too_long else f"ends with {title[-1]!r}"
            findings.append(
                Finding(
                    category="structural",
                    rule_id="sentence-header",
                    start=m.start(2),
                    end=m.end(2),
                    matched_text=title,
                    message=f"Header reads as a sentence ({reason}); headers are labels",
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
        """Three or more synonyms from one cluster within a sliding window.

        Cycling means using interchangeable terms for the *same* referent;
        terms spread over a whole document usually refer to different things
        (a grant proposal legitimately says "tool", "system" and "technology").
        """
        chunks = [text[s.start : s.end] for s in scannable_ranges(text, fmt)]
        words = [w.lower() for c in chunks for w in _REP_WORD.findall(c)]
        if len(words) < MIN_WORDS_FOR_SYN_CLUSTERS:
            return []
        term_patterns = {
            term: re.compile(rf"\b{re.escape(term)}s?\b")
            for cluster in _SYNONYM_CLUSTERS
            for term in cluster
        }
        findings: list[Finding] = []
        cluster_terms: dict[int, list[int]] = {}
        for ci, cluster in enumerate(_SYNONYM_CLUSTERS):
            positions: dict[str, list[int]] = {}
            for i, w in enumerate(words):
                for term in cluster:
                    if term_patterns[term].fullmatch(w):
                        positions.setdefault(term, []).append(i)
            cluster_terms[ci] = [p for ps in positions.values() for p in ps]
        for ci, cluster in enumerate(_SYNONYM_CLUSTERS):
            all_positions = sorted(cluster_terms[ci])
            if len(all_positions) < 3:
                continue
            # sliding window over word indices
            best: tuple[int, set[str]] = (0, set())
            lo = 0
            for hi, p in enumerate(all_positions):
                while p - all_positions[lo] > SYNONYM_WINDOW_WORDS:
                    lo += 1
                window = all_positions[lo : hi + 1]
                distinct_terms = {
                    t for t in cluster if any(term_patterns[t].fullmatch(words[q]) for q in window)
                }
                if len(distinct_terms) > len(best[1]):
                    best = (p, distinct_terms)
            if len(best[1]) >= 3:
                findings.append(
                    Finding(
                        category="structural",
                        rule_id="synonym-cycling",
                        start=0,
                        end=len(text),
                        matched_text=", ".join(sorted(best[1])),
                        message=(
                            f"Synonym cycling: {len(best[1])} interchangeable terms "
                            f"({', '.join(sorted(best[1]))}) within {SYNONYM_WINDOW_WORDS} words"
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

"""Rule-based (default) and optional LLM-assisted remediation of LLMisms."""

from __future__ import annotations

from dataclasses import dataclass, field

from ._llm import LLMError, rewrite_spans
from .detector import Finding

__all__ = ["Remediator", "RemediationResult"]


@dataclass
class RemediationResult:
    """Outcome of :meth:`Remediator.fix`."""

    text: str
    fixed: list[Finding] = field(default_factory=list)
    """Findings whose span was rewritten deterministically."""
    llm_fixed: list[Finding] = field(default_factory=list)
    """Findings rewritten by the LLM (only with ``use_llm=True``)."""
    remaining: list[Finding] = field(default_factory=list)
    """Findings left for the user to fix by hand."""
    errors: list[str] = field(default_factory=list)


class Remediator:
    """Applies fixes for detected findings, right-to-left to keep offsets valid."""

    def fix(
        self,
        text: str,
        findings: list[Finding],
        *,
        use_llm: bool = False,
        model: str | None = None,
    ) -> RemediationResult:
        result = RemediationResult(text=text)
        deterministic = [f for f in findings if f.suggestion is not None]
        structural = [f for f in findings if f.suggestion is None]

        out = text
        # Deduplicate overlapping same-suggestion fixes; apply right-to-left.
        applied: set[tuple[int, int, str]] = set()
        for f in sorted(deterministic, key=lambda f: (-f.end, f.start)):
            key = (f.start, f.end, f.suggestion or "")
            if key in applied:
                continue
            applied.add(key)
            start, end = (f.start, f.end) if f.suggestion else self._deletion_span(f, out)
            out = out[:start] + self._substitute(f, out) + out[end:]
            result.fixed.append(f)

        result.text = out
        if not structural:
            return result

        if not use_llm:
            result.remaining.extend(structural)
            return result

        # Re-map structural spans onto the rewritten text is unsafe; instead send
        # them keyed by matched text + original offsets for the LLM pass, then apply
        # longest-first matched-text replacement.
        try:
            spans = [(f.start, f.end, f.matched_text) for f in structural]
            rewrites = rewrite_spans(text=out, spans=spans, model=model)
        except LLMError as e:
            result.errors.append(str(e))
            result.remaining.extend(structural)
            return result

        for idx, f in enumerate(structural):
            replacement = rewrites.get(idx)
            if replacement is None or replacement == f.matched_text:
                result.remaining.append(f)
                continue
            result.text = result.text.replace(f.matched_text, replacement, 1)
            result.llm_fixed.append(f)
        return result

    @staticmethod
    def _deletion_span(f: Finding, text: str) -> tuple[int, int]:
        """Widen a pure deletion by one side's blanks so no double space remains.

        Blanks before the span go when it ends a word run (``a 🚀.``, ``a 🚀 b``);
        otherwise blanks after it go (``🚀 Deploy`` at line or bullet start).
        """
        start, end = f.start, f.end
        before = start
        while before > 0 and text[before - 1] in " \t":
            before -= 1
        after = end
        while after < len(text) and text[after] in " \t":
            after += 1
        at_line_start = before == 0 or text[before - 1] == "\n"
        if before < start and not at_line_start:
            return before, end
        return start, after

    @staticmethod
    def _substitute(f: Finding, text: str) -> str:
        """Apply the finding's suggestion, preserving initial capitalisation."""
        original = text[f.start : f.end]
        suggestion = original if f.suggestion is None else f.suggestion
        if original[:1].isupper() and suggestion[:1].islower():
            return suggestion[:1].upper() + suggestion[1:]
        return suggestion

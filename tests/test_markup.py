from __future__ import annotations

import pytest

from llmism._markup import bold_lead_in_bullets, scannable_ranges, split_sentences
from llmism._patterns import compiled_patterns, load_patterns


class TestPatternsData:
    def test_seed_list_is_curated_size(self) -> None:
        pats = load_patterns()
        assert 20 <= len(pats) <= 40

    def test_unique_ids_and_valid_categories(self) -> None:
        pats = load_patterns()
        ids = [p.id for p in pats]
        assert len(ids) == len(set(ids))
        assert {p.category for p in pats} <= {"lexical", "phrasal"}
        assert all(p.regex.flags for p in pats)  # compiled

    def test_compiled_patterns_cached(self) -> None:
        assert compiled_patterns() is compiled_patterns()


class TestScannableRanges:
    def test_text_whole_doc(self) -> None:
        ranges = scannable_ranges("hello world", "text")
        assert len(ranges) == 1
        assert ranges[0].start == 0 and ranges[0].end == 11

    def test_markdown_excludes_fenced_code(self) -> None:
        text = "before\n```python\ncode here\n```\nafter"
        ranges = scannable_ranges(text, "markdown")
        covered = "".join(text[s.start : s.end] for s in ranges)
        assert "code here" not in covered
        assert "before" in covered and "after" in covered

    def test_unterminated_fence_excluded_to_end(self) -> None:
        text = "intro\n```\ndelve\n"
        ranges = scannable_ranges(text, "markdown")
        covered = "".join(text[s.start : s.end] for s in ranges)
        assert "delve" not in covered

    def test_inline_code_excluded(self) -> None:
        text = "a `delve` b"
        ranges = scannable_ranges(text, "markdown")
        covered = "".join(text[s.start : s.end] for s in ranges)
        assert "delve" not in covered

    def test_latex_excludes_equation(self) -> None:
        text = "text\n\\begin{equation}\nx=1\n\\end{equation}\nmore"
        ranges = scannable_ranges(text, "latex")
        covered = "".join(text[s.start : s.end] for s in ranges)
        assert "x=1" not in covered
        assert "more" in covered

    def test_latex_excludes_inline_math(self) -> None:
        text = "value $x=1$ here"
        ranges = scannable_ranges(text, "latex")
        covered = "".join(text[s.start : s.end] for s in ranges)
        assert "x=1" not in covered
        assert "here" in covered


class TestSplitSentences:
    def test_offsets_point_into_original(self) -> None:
        text = "One sentence. Two sentences here! Three?"
        for sent, start, end in split_sentences(text, "text"):
            assert text[start:end] == sent

    def test_splits_on_all_terminators(self) -> None:
        sents = [s for s, _a, _b in split_sentences("A. B! C?", "text")]
        assert sents == ["A.", "B!", "C?"]

    def test_skips_code_fences(self) -> None:
        text = "Before.\n```python\nx = 1. y = 2\n```\nAfter."
        sents = [s for s, _a, _b in split_sentences(text, "markdown")]
        assert "Before." in sents and "After." in sents
        assert not any("x = 1" in s for s in sents)

    def test_does_not_split_decimals(self) -> None:
        sents = split_sentences("Version 3.5 shipped today.", "text")
        assert [s for s, _a, _b in sents] == ["Version 3.5 shipped today."]


class TestBoldLeadInBullets:
    def test_run_of_three_detected(self) -> None:
        text = "- **A:** x\n- **B:** y\n- **C:** z\n"
        spans = bold_lead_in_bullets(text, "markdown")
        assert len(spans) == 1
        assert "**A:**" in text[spans[0].start : spans[0].end]

    def test_two_not_enough(self) -> None:
        assert bold_lead_in_bullets("- **A:** x\n- **B:** y\n", "markdown") == []

    def test_numbered_list_counts(self) -> None:
        text = "1. **A:** x\n2. **B:** y\n3. **C:** z\n"
        assert len(bold_lead_in_bullets(text, "markdown")) == 1

    def test_latex_never_flags(self) -> None:
        text = "- **A:** x\n- **B:** y\n- **C:** z\n"
        assert bold_lead_in_bullets(text, "latex") == []


@pytest.mark.parametrize(
    ("word", "expected"),
    [
        ("delve", "delve"),
        ("delving", "delving"),
        ("delved", "delved"),
        ("boundaries", "boundaries"),
        ("Furthermore,", "Furthermore"),
        ("moreover", "moreover"),
    ],
)
def test_seed_patterns_fire(word: str, expected: str) -> None:
    det = __import__("llmism.detector", fromlist=["Detector"]).Detector()
    hits = [
        f
        for f in det.scan(f"Something {word} happened.", "text")
        if f.category in ("lexical", "phrasal")
    ]
    assert expected in " ".join(f.matched_text for f in hits)

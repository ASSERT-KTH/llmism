from __future__ import annotations

import pytest

from llmism.detector import Detector, Finding


@pytest.fixture()
def detector() -> Detector:
    return Detector()


class TestLexical:
    def test_delve_flagged_with_suggestion(self, detector: Detector) -> None:
        findings = detector.scan("We delve into the data.", "text")
        assert [f.rule_id for f in findings] == ["delve"]
        assert findings[0].category == "lexical"
        assert findings[0].suggestion == "explore"

    def test_clean_paragraph_has_no_lexical_findings(self, detector: Detector) -> None:
        text = "We looked at the data and found two errors. One was a typo."
        assert [f for f in detector.scan(text, "text") if f.category == "lexical"] == []

    def test_word_boundary_prevents_substring_match(self, detector: Detector) -> None:
        # "boundaries" must not match inside "aboundlessly"-style noise
        findings = detector.scan("The delimited region was abound.", "text")
        assert "boundaries" not in [f.rule_id for f in findings]

    def test_capitalised_match(self, detector: Detector) -> None:
        findings = detector.scan("Delve deeper.", "text")
        assert findings[0].matched_text == "Delve"

    def test_case_insensitive_offsets_are_exact(self, detector: Detector) -> None:
        text = "It will LEVERAGE the API."
        f = detector.scan(text, "text")[0]
        assert text[f.start : f.end] == "LEVERAGE"


class TestPhrasal:
    def test_its_not_x_its_y_flag_only(self, detector: Detector) -> None:
        findings = detector.scan("It's not a bug, it's a feature.", "text")
        assert [f.rule_id for f in findings] == ["its-not-x-its-y"]
        assert findings[0].category == "phrasal"
        assert findings[0].suggestion is None  # flagged, not auto-fixed

    def test_not_only_but_also(self, detector: Detector) -> None:
        findings = detector.scan("It is not only fast but also cheap.", "text")
        assert "not-only-but-also" in [f.rule_id for f in findings]

    def test_benefits_risks_trope(self, detector: Detector) -> None:
        text = "While this approach has benefits, it also carries risks."
        assert "has-benefits-carries-risks" in [f.rule_id for f in detector.scan(text, "text")]

    def test_normal_prose_not_flagged(self, detector: Detector) -> None:
        text = "The tool found two bugs and reported them. We fixed both today."
        assert [f for f in detector.scan(text, "text") if f.category == "phrasal"] == []


class TestStructural:
    def test_em_dash_density(self, detector: Detector) -> None:
        text = (
            "This is a long paragraph about writing style and its many problems — "
            "mostly caused by habit — that creep into drafts. Editors see this a lot — "
            "far more than they would like — and it makes prose feel machine written. "
            "Some writers never notice — they just keep going."
        )
        findings = [f for f in Detector().scan(text, "text") if f.rule_id == "em-dash-density"]
        assert len(findings) == 1
        assert findings[0].category == "structural"

    def test_few_em_dashes_not_flagged(self, detector: Detector) -> None:
        text = (
            "This is a long and otherwise ordinary paragraph about writing tools. "
            "It contains plenty of words and only one dash — which is fine. "
            "Nothing here should trip the density rule at all."
        )
        assert "em-dash-density" not in [f.rule_id for f in Detector().scan(text, "text")]

    def test_bold_lead_in_bullets(self, detector: Detector) -> None:
        text = (
            "Intro line.\n\n- **Performance:** fast.\n- **Clarity:** clear.\n- **Cost:** cheap.\n"
        )
        findings = [
            f for f in Detector().scan(text, "markdown") if f.rule_id == "bold-lead-in-bullets"
        ]
        assert len(findings) == 1

    def test_mixed_bullets_not_flagged(self, detector: Detector) -> None:
        text = "- **Performance:** fast.\n- plain item\n- **Cost:** cheap.\n"
        assert "bold-lead-in-bullets" not in [f.rule_id for f in Detector().scan(text, "markdown")]

    def test_rhetorical_question_answered(self, detector: Detector) -> None:
        text = "Why does this matter? The answer is simple: readers notice."
        findings = [
            f for f in Detector().scan(text, "text") if f.rule_id == "rhetorical-question-answer"
        ]
        assert len(findings) == 1

    def test_plain_question_not_flagged(self, detector: Detector) -> None:
        text = "Did you run the tests? They passed on my machine."
        assert "rhetorical-question-answer" not in [
            f.rule_id for f in Detector().scan(text, "text")
        ]

    def test_low_burstiness_whole_doc(self, detector: Detector) -> None:
        text = (
            "The system parses the input text carefully. "
            "The detector scans for known patterns. "
            "The remediator applies safe fixes. "
            "The report lists remaining issues. "
            "The user reviews each flagged item."
        )
        findings = [f for f in Detector().scan(text, "text") if f.rule_id == "low-burstiness"]
        assert len(findings) == 1
        assert findings[0].start == 0
        assert findings[0].end == len(text)

    def test_varied_prose_not_flagged_for_burstiness(self, detector: Detector) -> None:
        text = (
            "Short. "
            "This second sentence is considerably longer than the first one, winding on. "
            "Medium length again here. "
            "Longer still, the fourth sentence adds subordinate clauses that stretch it out "
            "well beyond the average, deliberately so. "
            "Brief."
        )
        assert "low-burstiness" not in [f.rule_id for f in Detector().scan(text, "text")]


class TestFormats:
    def test_markdown_code_fence_ignored(self, detector: Detector) -> None:
        text = "Clean intro.\n\n```python\ndelve into boundaries\n```\n\nClean outro.\n"
        assert detector.scan(text, "markdown") == []

    def test_inline_code_ignored(self, detector: Detector) -> None:
        assert detector.scan("Use `delve` carefully.", "markdown") == []

    def test_latex_math_ignored(self, detector: Detector) -> None:
        text = (
            "Clean text here.\n\\begin{equation}\ndelve = leverage\n\\end{equation}\n"
            "More clean text $delve$ inline.\n"
        )
        assert detector.scan(text, "latex") == []

    def test_latex_verbatim_ignored(self, detector: Detector) -> None:
        text = "\\begin{verbatim}\ndelve deep\n\\end{verbatim}\n"
        assert detector.scan(text, "latex") == []

    def test_plain_text_format_scans_everything(self, detector: Detector) -> None:
        assert [f.rule_id for f in Detector().scan("delve", "text")] == ["delve"]


class TestFindingShape:
    def test_finding_fields(self, detector: Detector) -> None:
        (f,) = detector.scan("It delves deep.", "text")
        assert isinstance(f, Finding)
        assert f.start < f.end
        assert "delv" in f.matched_text
        assert f.message


class TestLoadBearing:
    def test_load_bearing_flagged(self, detector: Detector) -> None:
        (f,) = detector.scan("This is a load-bearing dependency.", "text")
        assert f.rule_id == "load-bearing"
        assert f.suggestion == "essential"

    def test_hyphenated_and_spaced_variants(self, detector: Detector) -> None:
        ids = [f.rule_id for f in detector.scan("a load-bearing wall; a load bearing role", "text")]
        assert ids.count("load-bearing") == 2

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
    def test_em_dash_overuse(self, detector: Detector) -> None:
        text = (
            "This is a long paragraph about writing style and its many problems — "
            "mostly caused by habit — that creep into drafts. Editors see this a lot — "
            "far more than they would like — and it makes prose feel machine written. "
            "Some writers never notice — they just keep going."
        )
        findings = [f for f in Detector().scan(text, "text") if f.rule_id == "em-dash-overuse"]
        assert len(findings) == 1
        assert findings[0].category == "structural"
        assert "5 em-dashes" in findings[0].message

    def test_em_dashes_in_separate_paragraphs_not_flagged(self, detector: Detector) -> None:
        text = (
            "First paragraph with one dash — only one.\n\n"
            "Second paragraph with one dash — also fine.\n\n"
            "Third paragraph with one dash — still fine.\n"
        )
        assert "em-dash-overuse" not in [f.rule_id for f in detector.scan(text, "text")]

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


class TestNewStructuralRules:
    def test_transition_cluster(self, detector: Detector) -> None:
        text = (
            "The team launched the product in Q1. Additionally, they expanded to "
            "three new markets. Furthermore, satisfaction improved. Moreover, churn "
            "dropped below five percent."
        )
        (f,) = [f for f in detector.scan(text, "text") if f.rule_id == "transition-cluster"]
        assert "Additionally" in f.message and "Moreover" in f.message

    def test_single_transition_not_flagged(self, detector: Detector) -> None:
        text = "One sentence here. Furthermore, a second one follows it now."
        assert "transition-cluster" not in [f.rule_id for f in detector.scan(text, "text")]

    def test_transition_cluster_scoped_to_paragraph(self, detector: Detector) -> None:
        text = "Para one uses Additionally here.\n\nPara two uses Furthermore instead.\n"
        assert "transition-cluster" not in [f.rule_id for f in detector.scan(text, "text")]

    def test_opener_repetition(self, detector: Detector) -> None:
        text = (
            "The client retries twice. The client waits longer each time. "
            "The client gives up after thirty seconds and raises a timeout."
        )
        (f,) = [f for f in detector.scan(text, "text") if f.rule_id == "opener-repetition"]
        assert "the client" in f.message

    def test_varied_openers_not_flagged(self, detector: Detector) -> None:
        text = (
            "Birds fly south. In autumn they leave. Some stay behind and "
            "perish quietly in the cold northern winter."
        )
        assert "opener-repetition" not in [f.rule_id for f in detector.scan(text, "text")]

    def test_hedge_stacking(self, detector: Detector) -> None:
        text = "The change could potentially reduce errors somewhat, and might perhaps help."
        (f,) = [f for f in detector.scan(text, "text") if f.rule_id == "hedge-stacking"]
        assert "might" in f.message and "potentially" in f.message

    def test_single_hedge_not_flagged(self, detector: Detector) -> None:
        assert "hedge-stacking" not in [
            f.rule_id for f in detector.scan("This may help a little.", "text")
        ]

    def test_uniform_paragraphs(self, detector: Detector) -> None:
        text = "\n\n".join(" ".join(f"Word{i} here." for i in range(4)) for _ in range(5))
        assert "uniform-paragraphs" in [f.rule_id for f in detector.scan(text, "text")]

    def test_varied_paragraphs_not_flagged(self, detector: Detector) -> None:
        text = (
            "One.\n\n"
            + " ".join(f"Sentence {i} is longer than the last one here." for i in range(6))
            + "\n\nTwo words only.\n\n"
            + " ".join(f"Another {i}." for i in range(5))
        )
        assert "uniform-paragraphs" not in [f.rule_id for f in detector.scan(text, "text")]

    def test_mechanical_cadence(self, detector: Detector) -> None:
        text = (
            "It works. The design combines three separate caching layers into one "
            "pipeline that scales. It ships. Free tier users get the same "
            "performance improvements as enterprise customers do today. "
            "It helps. Nobody expected the second release to land this quickly at all."
        )
        assert "mechanical-cadence" in [f.rule_id for f in detector.scan(text, "text")]

    def test_synonym_cycling(self, detector: Detector) -> None:
        text = (
            "The company was founded in a shed. The firm took an apprentice in 1974. "
            "By 1990 the organization employed sixty people, and the business was "
            "casting for the whole county around it. This is filler to reach forty "
            "words so the cluster check engages properly here now."
        )
        (f,) = [f for f in detector.scan(text, "text") if f.rule_id == "synonym-cycling"]
        assert "company" in f.message

    def test_short_text_skips_synonym_cycling(self, detector: Detector) -> None:
        text = "The company and the firm met the organization."
        assert "synonym-cycling" not in [f.rule_id for f in detector.scan(text, "text")]

    def test_degenerate_repetition(self, detector: Detector) -> None:
        phrase = "a shortage of affordable housing is caused in part by a lack of supply "
        text = phrase * 6
        (f,) = [f for f in detector.scan(text, "text") if f.rule_id == "degenerate-repetition"]
        assert "6-word spans repeat" in f.message

    def test_normal_prose_not_degenerate(self, detector: Detector) -> None:
        text = (
            "The pipeline records vectors with their symbolic sources. "
            "Each record carries the outcome of the execution that produced it. "
            "Analysts query the staging schema through dashboards they already use. "
            "Every night the job copies events and drops duplicates against a window. "
            "Clean rows land in reporting by the following morning at nine sharp."
        )
        assert "degenerate-repetition" not in [f.rule_id for f in detector.scan(text, "text")]


class TestNewPhrasalRules:
    def test_ing_tail(self, detector: Detector) -> None:
        text = "The canopies were cast a mile away, underscoring its railway heritage."
        assert "ing-tail" in [f.rule_id for f in detector.scan(text, "text")]

    def test_factual_ing_tail_not_flagged(self, detector: Detector) -> None:
        text = "The cache was cleared, resulting in faster builds for everyone."
        assert "ing-tail" not in [f.rule_id for f in detector.scan(text, "text")]

    def test_vague_attribution(self, detector: Detector) -> None:
        assert "vague-attribution" in [
            f.rule_id for f in detector.scan("Studies suggest a link.", "text")
        ]

    def test_catalog_leadin(self, detector: Detector) -> None:
        assert "catalog-leadin" in [
            f.rule_id for f in detector.scan("It uses several methods here.", "text")
        ]

    def test_catalog_pivot(self, detector: Detector) -> None:
        assert "catalog-pivot" in [
            f.rule_id for f in detector.scan("These methods give the team headroom.", "text")
        ]

    def test_empty_pivot(self, detector: Detector) -> None:
        assert "empty-pivot" in [
            f.rule_id for f in detector.scan("It's worth noting that it returns null.", "text")
        ]

    def test_simple_yet(self, detector: Detector) -> None:
        assert "simple-yet" in [
            f.rule_id for f in detector.scan("The tool is simple yet powerful.", "text")
        ]

    def test_significance_inflation(self, detector: Detector) -> None:
        assert "significance-inflation" in [
            f.rule_id for f in detector.scan("It stands as a testament to his work.", "text")
        ]

    def test_synonym_cycling_requires_proximity(self, detector: Detector) -> None:
        """Same-cluster terms far apart usually name different referents."""
        text = (
            "Our company ships one tool for analysts working on market data. "
            + " ".join(
                f"Filler sentence number {i} talks about ordinary topics." for i in range(60)
            )
            + " Decades later a separate system was built and the underlying technology matured."
        )
        assert "synonym-cycling" not in [f.rule_id for f in detector.scan(text, "text")]

    def test_synonym_cycling_same_referent(self, detector: Detector) -> None:
        text = (
            "The firm shipped early and often. The company grew fast afterwards. "
            "The organization hired many people that year. The business then "
            "plateaued for a year before restarting its slow expansion into two "
            "new regional markets abroad, which took considerable time overall."
        )
        (f,) = [f for f in detector.scan(text, "text") if f.rule_id == "synonym-cycling"]
        assert "company" in f.message

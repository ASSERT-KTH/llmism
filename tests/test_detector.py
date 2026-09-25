from __future__ import annotations

import pytest

from llmism.detector import Detector, Finding


@pytest.fixture()
def detector() -> Detector:
    return Detector()


class TestLexical:
    def test_leverage_flagged_with_suggestion(self, detector: Detector) -> None:
        findings = detector.scan("We leverage the data.", "text")
        assert [f.rule_id for f in findings] == ["leverage-verb"]
        assert findings[0].category == "lexical"
        assert findings[0].suggestion == "use"

    def test_merely_flagged_without_auto_fix(self, detector: Detector) -> None:
        findings = detector.scan("This is merely a starting point.", "text")
        assert [f.rule_id for f in findings] == ["merely"]
        assert findings[0].category == "lexical"
        assert findings[0].suggestion is None

    def test_clean_paragraph_has_no_lexical_findings(self, detector: Detector) -> None:
        text = "We looked at the data and found two errors. One was a typo."
        assert [f for f in detector.scan(text, "text") if f.category == "lexical"] == []

    def test_word_boundary_prevents_substring_match(self, detector: Detector) -> None:
        # "boundaries" must not match inside "aboundlessly"-style noise
        findings = detector.scan("The delimited region was abound.", "text")
        assert "boundaries" not in [f.rule_id for f in findings]

    def test_capitalised_match(self, detector: Detector) -> None:
        findings = detector.scan("Merely a start.", "text")
        assert findings[0].matched_text == "Merely"

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

    def test_rather_than_flagged_without_auto_fix(self, detector: Detector) -> None:
        findings = detector.scan("Use a local cache rather than the network.", "text")
        assert [f.rule_id for f in findings] == ["rather-than"]
        assert findings[0].suggestion is None

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
        text = "Clean intro.\n\n```python\nleverage into boundaries\n```\n\nClean outro.\n"
        assert detector.scan(text, "markdown") == []

    def test_inline_code_ignored(self, detector: Detector) -> None:
        assert detector.scan("Use `leverage` carefully.", "markdown") == []

    def test_latex_math_ignored(self, detector: Detector) -> None:
        text = (
            "Clean text here.\n\\begin{equation}\nleverage = leverage\n\\end{equation}\n"
            "More clean text $leverage$ inline.\n"
        )
        assert detector.scan(text, "latex") == []

    def test_latex_verbatim_ignored(self, detector: Detector) -> None:
        text = "\\begin{verbatim}\nleverage deep\n\\end{verbatim}\n"
        assert detector.scan(text, "latex") == []

    def test_plain_text_format_scans_everything(self, detector: Detector) -> None:
        assert [f.rule_id for f in Detector().scan("leverage", "text")] == ["leverage-verb"]


class TestFindingShape:
    def test_finding_fields(self, detector: Detector) -> None:
        (f,) = detector.scan("It leverages deep.", "text")
        assert isinstance(f, Finding)
        assert f.start < f.end
        assert "leverag" in f.matched_text
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

    def test_colon_reveal(self, detector: Detector) -> None:
        assert "colon-reveal" in [
            f.rule_id for f in detector.scan("Here's the thing, it returns null.", "text")
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

    def test_genuine_variants(self, detector: Detector) -> None:
        assert [f.rule_id for f in detector.scan("a genuine gain", "text")] == ["genuine"]
        assert [f.rule_id for f in detector.scan("genuinely different", "text")] == ["genuine"]
        assert "genuine" not in [f.rule_id for f in detector.scan("ingenuous", "text")]

    # -- rules imported from claude-style-patch STYLE.md ----------------------
    def test_colon_clause(self, detector: Detector) -> None:
        text = "The honest construction here: the parser never sees those bytes."
        assert "colon-clause" in [f.rule_id for f in detector.scan(text, "text")]

    def test_colon_clause_allows_lists_and_labels(self, detector: Detector) -> None:
        listed = "It ships three parts: a detector, a remediator, and a CLI."
        assert "colon-clause" not in [f.rule_id for f in detector.scan(listed, "text")]
        label = "Note: the cache is rebuilt on every run."
        assert "colon-clause" not in [f.rule_id for f in detector.scan(label, "text")]

    def test_colon_clause_skips_urls_and_headings(self, detector: Detector) -> None:
        text = "# Why this matters: a longer story\n\nSee https://example.org/a/b for details.\n"
        assert "colon-clause" not in [f.rule_id for f in detector.scan(text)]

    def test_verbless_fragment(self, detector: Detector) -> None:
        text = "Two things worth watching. The first is whether it holds up."
        assert "verbless-fragment" in [f.rule_id for f in detector.scan(text, "text")]

    def test_verbless_fragment_ignores_real_sentences(self, detector: Detector) -> None:
        text = "The parser is slow. One caution applied here."
        assert "verbless-fragment" not in [f.rule_id for f in detector.scan(text, "text")]

    def test_sentence_header(self, detector: Detector) -> None:
        text = "# The parser rewrites offsets from right to left always.\n\nBody text here.\n"
        rules = [f.rule_id for f in detector.scan(text)]
        assert "sentence-header" in rules

    def test_sentence_header_allows_labels(self, detector: Detector) -> None:
        assert "sentence-header" not in [f.rule_id for f in detector.scan("## Installation\n\nx\n")]

    def test_latinate_suggestions(self, detector: Detector) -> None:
        (f,) = [
            f
            for f in detector.scan("Approximately ten runs.", "text")
            if f.rule_id == "approximately"
        ]
        assert f.suggestion == "about"

    def test_depth_signaling_and_announcing(self, detector: Detector) -> None:
        rules = [f.rule_id for f in detector.scan("At a more fundamental level, x.", "text")]
        assert "depth-signaling" in rules
        rules = [f.rule_id for f in detector.scan("The key insight is that x.", "text")]
        assert "announcing-label" in rules

    def test_colon_clause_skips_code_lines(self, detector: Detector) -> None:
        text = "byte0=0x80: low7 is zero so it continues.\nLet me map the positions here."
        assert "colon-clause" not in [f.rule_id for f in detector.scan(text, "text")]

    def test_sentence_header_skips_code_fences(self, detector: Detector) -> None:
        text = "```sh\n# 2. the real test is to swap the cable and retry\nls\n```\n"
        assert "sentence-header" not in [f.rule_id for f in detector.scan(text)]


class TestAvoidAiWritingPatterns:
    @pytest.mark.parametrize(
        ("text", "rule_id"),
        [
            ("Certainly! Here is the report.", "chatbot-opener"),
            ("Experts believe it will work.", "vague-attribution"),
            ("Only time will tell.", "generic-conclusion"),
            ("This marks a watershed moment.", "significance-inflation"),
            ("Sign it [Your Name].", "unfilled-placeholder"),
            ("Filed on 2025-XX-XX.", "unfilled-placeholder"),
            ("See citeturn0search0.", "chatbot-citation-markup"),
            ("See contentReference[oaicite:0].", "chatbot-citation-markup"),
            ("Visit https://example.com/?utm_source=chatgpt.com", "ai-tool-url-parameter"),
            ("The log serves as evidence.", "serves-as"),
        ],
    )
    def test_new_patterns(self, detector: Detector, text: str, rule_id: str) -> None:
        matches = [f for f in detector.scan(text, "text") if f.rule_id == rule_id]
        assert len(matches) == 1
        assert text[matches[0].start : matches[0].end] == matches[0].matched_text
        assert matches[0].suggestion is None

    @pytest.mark.parametrize(
        ("text", "rule_id", "suggestion"),
        [
            ("In order to start, press enter.", "in-order-to", "to"),
            ("We stopped due to the fact that it rained.", "due-to-the-fact-that", "because"),
        ],
    )
    def test_safe_clarity_fixes(
        self, detector: Detector, text: str, rule_id: str, suggestion: str
    ) -> None:
        (finding,) = [f for f in detector.scan(text, "text") if f.rule_id == rule_id]
        assert finding.suggestion == suggestion

    def test_chatbot_opener_only_at_start_of_line(self, detector: Detector) -> None:
        assert "chatbot-opener" not in [
            f.rule_id for f in detector.scan("She said, 'Absolutely!' and left.", "text")
        ]

    def test_protected_markup_is_not_flagged(self, detector: Detector) -> None:
        text = (
            "```text\nCertainly! [Your Name] citeturn0search0\n"
            "#One #Two #Three #Four #Five #Six\n```"
        )
        assert detector.scan(text, "markdown") == []

    def test_hashtag_stuffing(self, detector: Detector) -> None:
        text = "A post.\n#One #Two #Three #Four #Five #Six\n"
        (finding,) = [f for f in detector.scan(text, "markdown") if f.rule_id == "hashtag-stuffing"]
        assert finding.matched_text.strip() == "#One #Two #Three #Four #Five #Six"

    def test_five_hashtags_and_headings_are_allowed(self, detector: Detector) -> None:
        text = "# A heading with six ordinary words here\n#One #Two #Three #Four #Five\n"
        assert "hashtag-stuffing" not in [f.rule_id for f in detector.scan(text)]

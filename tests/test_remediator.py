from __future__ import annotations

import pytest

from llmism.detector import Detector
from llmism.remediator import Remediator


@pytest.fixture()
def remediator() -> Remediator:
    return Remediator()


class TestDeterministic:
    def test_lexical_replacement(self, remediator: Remediator) -> None:
        text = "We delve into the data."
        findings = Detector().scan(text, "text")
        result = remediator.fix(text, findings)
        assert result.text == "We explore into the data."
        assert [f.rule_id for f in result.fixed] == ["delve"]
        assert result.remaining == []

    def test_capitalisation_preserved(self, remediator: Remediator) -> None:
        text = "Delve deeper."
        findings = Detector().scan(text, "text")
        result = remediator.fix(text, findings)
        assert result.text == "Explore deeper."

    def test_multiple_fixes_offsets_stay_valid(self, remediator: Remediator) -> None:
        text = "We delve into it and leverage the tool. It also has boundaries."
        findings = Detector().scan(text, "text")
        result = remediator.fix(text, findings)
        assert "delve" not in result.text
        assert "leverage" not in result.text
        assert "boundaries" not in result.text
        assert "use the tool" in result.text
        assert len(result.fixed) == 3

    def test_flag_only_phrases_left_untouched(self, remediator: Remediator) -> None:
        text = "It's not a bug, it's a feature."
        findings = Detector().scan(text, "text")
        result = remediator.fix(text, findings)
        assert result.text == text
        assert [f.rule_id for f in result.remaining] == ["its-not-x-its-y"]

    def test_merely_left_untouched_without_auto_fix(self, remediator: Remediator) -> None:
        text = "This is merely a starting point."
        result = remediator.fix(text, Detector().scan(text, "text"))
        assert result.text == text
        assert [f.rule_id for f in result.remaining] == ["merely"]

    def test_more_than_just_left_untouched_without_auto_fix(self, remediator: Remediator) -> None:
        text = "This is more than just a formatting tool."
        result = remediator.fix(text, Detector().scan(text, "text"))
        assert result.text == text
        assert [f.rule_id for f in result.remaining] == ["more-than-just"]

    def test_rather_than_left_untouched_without_auto_fix(self, remediator: Remediator) -> None:
        text = "Use a local cache rather than the network."
        result = remediator.fix(text, Detector().scan(text, "text"))
        assert result.text == text
        assert [f.rule_id for f in result.remaining] == ["rather-than"]

    def test_structural_left_untouched_without_llm(self, remediator: Remediator) -> None:
        text = (
            "This is a long paragraph about style and its problems — mostly habit — "
            "that creep into drafts. Editors see this a lot — far more than they like — "
            "and it makes prose feel machine written. Some writers never notice — they "
            "just keep going."
        )
        findings = Detector().scan(text, "text")
        result = remediator.fix(text, findings)
        assert result.text == text
        assert any(f.rule_id == "em-dash-overuse" for f in result.remaining)
        assert result.llm_fixed == []


class TestLlmPath:
    def test_llm_rewrite_applied_with_mocked_client(
        self, remediator: Remediator, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        text = "Why use it? The answer is that it helps."

        class FakeBlock:
            type = "text"
            text = '{"0": "It helps, in short."}'

        class FakeMessages:
            def create(self, **kwargs: object) -> object:
                return type("Resp", (), {"content": [FakeBlock()]})()

        class FakeAnthropic:
            def __init__(self, *args: object, **kwargs: object) -> None:
                self.messages: object = FakeMessages()

        import sys
        import types

        fake_module = types.ModuleType("anthropic")
        fake_module.Anthropic = FakeAnthropic  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "anthropic", fake_module)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")

        findings = Detector().scan(text, "text")
        result = remediator.fix(text, findings, use_llm=True)
        assert "It helps, in short." in result.text
        assert any(f.rule_id == "rhetorical-question-answer" for f in result.llm_fixed)

    def test_llm_error_surfaced(
        self, remediator: Remediator, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        text = "Why use it? The answer is that it helps."
        findings = Detector().scan(text, "text")
        monkeypatch.setattr(
            "llmism.remediator.rewrite_spans",
            lambda **kwargs: (_ for _ in ()).throw(
                __import__("llmism._llm", fromlist=["LLMError"]).LLMError("api down")
            ),
        )
        result = remediator.fix(text, findings, use_llm=True)
        assert result.errors == ["api down"]
        assert result.text == text
        assert result.remaining


class TestEmptyAndClean:
    def test_no_findings_is_noop(self, remediator: Remediator) -> None:
        text = "Two bugs were found. Both were fixed."
        result = remediator.fix(text, Detector().scan(text, "text"))
        assert result.text == text
        assert result.fixed == result.remaining == result.llm_fixed == []

    def test_empty_text(self, remediator: Remediator) -> None:
        result = remediator.fix("", [])
        assert result.text == ""

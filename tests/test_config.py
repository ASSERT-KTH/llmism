from __future__ import annotations

from pathlib import Path

import pytest

from llmism._config import Config, ConfigError, load_config
from llmism.detector import Detector

EM_DASH_TEXT = (
    "This is a long paragraph about style and its problems — mostly habit — "
    "that creep into drafts. Editors see this a lot — far more than they like — "
    "and it makes prose feel machine written. Some writers never notice — they "
    "just keep going."
)


def write_config(tmp_path: Path, content: str) -> Path:
    p = tmp_path / ".llmism.toml"
    p.write_text(content, encoding="utf-8")
    return p


class TestLoadConfig:
    def test_missing_file_gives_defaults(self, tmp_path: Path) -> None:
        cfg = load_config(tmp_path / "nope.toml")
        assert cfg == Config()
        assert not cfg.ignored("em-dash-overuse")
        assert cfg.param("em-dash-overuse", "max-per-paragraph") == 2

    def test_ignore_list(self, tmp_path: Path) -> None:
        cfg = load_config(
            write_config(tmp_path, 'ignore = ["em-dash-overuse", "merely"]\n')
        )
        assert cfg.ignored("em-dash-overuse")
        assert cfg.ignored("merely")
        assert not cfg.ignored("leverage-verb")

    def test_threshold_override(self, tmp_path: Path) -> None:
        cfg = load_config(
            write_config(
                tmp_path,
                "[rules.em-dash-overuse]\nmax-per-paragraph = 4\n",
            )
        )
        assert cfg.param("em-dash-overuse", "max-per-paragraph") == 4
        assert cfg.param("hedge-stacking", "min") == 2  # untouched default

    def test_suggestion_override_rejected_for_structural(self, tmp_path: Path) -> None:
        # structural rules have no suggestion key
        with pytest.raises(ConfigError, match="unknown key 'suggestion'"):
            load_config(
                write_config(
                    tmp_path,
                    '[rules.em-dash-overuse]\nsuggestion = "x"\n',
                )
            )

    def test_unknown_rule_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="unknown rule"):
            load_config(write_config(tmp_path, "[rules.nope]\nmin = 1\n"))

    def test_unknown_key_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="unknown key"):
            load_config(
                write_config(tmp_path, "[rules.em-dash-overuse]\nmax-per-line = 4\n")
            )

    def test_unknown_top_level_key_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="unknown top-level key"):
            load_config(write_config(tmp_path, "select = ['all']\n"))

    def test_include_exclude_roundtrip(self, tmp_path: Path) -> None:
        cfg = load_config(
            write_config(
                tmp_path, 'include = [".rst"]\nexclude = ["vendor", "generated"]\n'
            )
        )
        assert cfg.include == frozenset({".rst"})
        assert cfg.exclude == frozenset({"vendor", "generated"})

    def test_include_not_a_list_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="include must be a list"):
            load_config(write_config(tmp_path, 'include = ".rst"\n'))

    def test_exclude_wrong_element_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="exclude must be a list"):
            load_config(write_config(tmp_path, "exclude = [1, 2]\n"))


class TestDetectorWithConfig:
    def test_ignored_rule_not_reported(self, tmp_path: Path) -> None:
        cfg = load_config(write_config(tmp_path, 'ignore = ["em-dash-overuse"]\n'))
        findings = Detector(config=cfg).scan(EM_DASH_TEXT, "text")
        assert all(f.rule_id != "em-dash-overuse" for f in findings)

    def test_loosened_threshold_stops_firing(self, tmp_path: Path) -> None:
        cfg = load_config(
            write_config(tmp_path, "[rules.em-dash-overuse]\nmax-per-paragraph = 6\n")
        )
        findings = Detector(config=cfg).scan(EM_DASH_TEXT, "text")
        assert all(f.rule_id != "em-dash-overuse" for f in findings)

    def test_tightened_threshold_still_fires(self, tmp_path: Path) -> None:
        cfg = load_config(
            write_config(tmp_path, "[rules.em-dash-overuse]\nmax-per-paragraph = 1\n")
        )
        findings = Detector(config=cfg).scan(EM_DASH_TEXT, "text")
        assert any(f.rule_id == "em-dash-overuse" for f in findings)

    def test_suggestion_override_changes_fix(self, tmp_path: Path) -> None:
        cfg = load_config(
            write_config(tmp_path, '[rules.leverage-verb]\nsuggestion = "draw on"\n'),
        )
        findings = Detector(config=cfg).scan("We leverage the data.", "text")
        assert findings[0].suggestion == "draw on"

    def test_thresholds_do_not_leak_across_instances(self) -> None:
        # default detector after a configured one must keep default thresholds
        import llmism.detector as d

        strict = Config(rules={"em-dash-overuse": {"max-per-paragraph": 1}})
        Detector(config=strict)
        assert d.EM_DASH_MAX_PER_PARAGRAPH == 1
        Detector()  # rebinding to defaults
        assert d.EM_DASH_MAX_PER_PARAGRAPH == 2


class TestCliConfig:
    def test_detect_honors_ignore(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        import json

        from llmism.cli import main

        write_config(tmp_path, 'ignore = ["leverage-verb"]\n')
        p = tmp_path / "doc.md"
        p.write_text("We leverage the data.", encoding="utf-8")
        rc = main(["detect", str(p), "--config", str(tmp_path / ".llmism.toml")])
        payload = json.loads(capsys.readouterr().out)
        assert payload == []
        assert rc == 0

    def test_bad_config_exits_two(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from llmism.cli import main

        write_config(tmp_path, "[rules.nope]\nmin = 1\n")
        p = tmp_path / "doc.md"
        p.write_text("We leverage the data.", encoding="utf-8")
        with pytest.raises(SystemExit) as e:
            main(["detect", str(p), "--config", str(tmp_path / ".llmism.toml")])
        assert e.value.code == 2
        assert "unknown rule" in capsys.readouterr().err

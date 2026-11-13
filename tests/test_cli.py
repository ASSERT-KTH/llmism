from __future__ import annotations

import json
import shlex
from pathlib import Path

import pytest

from llmism.cli import main


@pytest.fixture()
def sample_md(tmp_path: Path) -> Path:
    p = tmp_path / "sample.md"
    p.write_text(
        "# Notes\n\n"
        "We leverage the data and elevate the tool.\n\n"
        "It's not a bug, it's a feature.\n\n"
        "```python\nprint('leverage')\n```\n",
        encoding="utf-8",
    )
    return p


class TestDetect:
    def test_human_table_output(self, sample_md: Path, capsys: pytest.CaptureFixture[str]) -> None:
        rc = main(["detect", str(sample_md), "--human"])
        out = capsys.readouterr().out
        assert "leverage" in out and "its-not-x-its-y" in out
        assert "lexical" in out
        assert "print" not in out  # code fence skipped
        assert rc == 1

    def test_json_is_default(self, sample_md: Path, capsys: pytest.CaptureFixture[str]) -> None:
        main(["detect", str(sample_md)])
        payload = json.loads(capsys.readouterr().out)
        assert any(f["rule"] == "leverage-verb" for f in payload)
        assert all("start" in f and "category" in f for f in payload)

    def test_json_flag_still_works(
        self, sample_md: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        main(["detect", str(sample_md), "--json"])
        payload = json.loads(capsys.readouterr().out)
        assert any(f["rule"] == "leverage-verb" for f in payload)

    def test_fix_hint_included(self, sample_md: Path, capsys: pytest.CaptureFixture[str]) -> None:
        main(["detect", str(sample_md)])
        payload = json.loads(capsys.readouterr().out)
        by_rule = {f["rule"]: f["fix"] for f in payload}
        # deterministic fix: replacement + command
        det = by_rule["leverage-verb"]
        assert det["kind"] == "replace"
        assert det["replacement"] == "use"
        assert det["command"] == f"llmism fix {shlex.quote(str(sample_md))}"
        # flag-only finding: llm rewrite + hint
        llm = by_rule["its-not-x-its-y"]
        assert llm["kind"] == "llm"
        assert llm["command"] == f"llmism fix {shlex.quote(str(sample_md))} --llm"
        assert llm["hint"]

    def test_fix_hint_quotes_paths(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        p = tmp_path / "my draft.md"
        p.write_text("We leverage it.", encoding="utf-8")
        main(["detect", str(p)])
        payload = json.loads(capsys.readouterr().out)
        assert payload[0]["fix"]["command"] == f"llmism fix '{p}'"

    def test_stdin(
        self, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import io

        monkeypatch.setattr("sys.stdin", io.StringIO("We leverage."))
        main(["detect", "-", "--human"])
        assert "leverage" in capsys.readouterr().out

    def test_clean_file_returns_zero(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        p = tmp_path / "ok.txt"
        p.write_text("Two bugs were found. Both fixed.", encoding="utf-8")
        rc = main(["detect", str(p)])
        payload = json.loads(capsys.readouterr().out)
        assert payload == []
        assert rc == 0

    def test_format_override_for_stdin(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        import io

        monkeypatch.setattr("sys.stdin", io.StringIO("a `leverage` b"))
        rc = main(["detect", "-", "--format", "markdown"])
        payload = json.loads(capsys.readouterr().out)
        assert payload == []
        assert rc == 0

    def test_latex_extension_autodetected(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        p = tmp_path / "doc.tex"
        p.write_text(
            "text\n\\begin{equation}\nleverage=1\n\\end{equation}\n",
            encoding="utf-8",
        )
        rc = main(["detect", str(p)])
        assert rc == 0


class TestFix:
    def test_human_stdout_rewrite(
        self, sample_md: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        rc = main(["fix", str(sample_md), "--human"])
        out = capsys.readouterr().out
        assert "We use the data" in out  # prose rewritten
        assert "print('leverage')" in out  # code fence untouched
        assert "It's not a bug" in out  # flag-only pattern kept
        assert rc == 1  # remaining manual findings

    def test_json_is_default(self, sample_md: Path, capsys: pytest.CaptureFixture[str]) -> None:
        main(["fix", str(sample_md)])
        payload = json.loads(capsys.readouterr().out)
        assert isinstance(payload, list) and len(payload) == 1
        assert any(f["rule"] == "leverage-verb" for f in payload[0]["fixed"])
        assert any(f["rule"] == "its-not-x-its-y" for f in payload[0]["remaining"])
        assert "We use the data" in payload[0]["text"]

    def test_json_flag_still_works(
        self, sample_md: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        main(["fix", str(sample_md), "--json"])
        payload = json.loads(capsys.readouterr().out)
        assert isinstance(payload, list) and len(payload) == 1
        assert "We use the data" in payload[0]["text"]

    def test_in_place(self, sample_md: Path, capsys: pytest.CaptureFixture[str]) -> None:
        main(["fix", str(sample_md), "--in-place"])
        content = sample_md.read_text(encoding="utf-8")
        assert "We use the data" in content
        assert "print('leverage')" in content  # code fence untouched

    def test_clean_fix_exits_zero(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        p = tmp_path / "ok.md"
        p.write_text("Two bugs found. Both fixed.", encoding="utf-8")
        rc = main(["fix", str(p)])
        assert rc == 0

    def test_diff_prints_unified_diff(
        self, sample_md: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        rc = main(["fix", str(sample_md), "--diff"])
        out = capsys.readouterr().out
        assert f"--- a/{sample_md}" in out
        assert f"+++ b/{sample_md}" in out
        assert "-We leverage the data and elevate the tool." in out
        assert "+We use the data and improve the tool." in out
        assert "print('leverage')" not in out  # untouched lines not in diff body
        assert rc == 1  # remaining manual findings still gate the exit code

    def test_diff_clean_file_prints_nothing(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        p = tmp_path / "ok.md"
        p.write_text("Two bugs found. Both fixed.", encoding="utf-8")
        rc = main(["fix", str(p), "--diff"])
        assert capsys.readouterr().out == ""
        assert rc == 0

    def test_diff_overrides_human_text_output(
        self, sample_md: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        main(["fix", str(sample_md), "--diff", "--human"])
        out = capsys.readouterr().out
        assert out.startswith("---")  # diff, not the rewritten prose
        assert "need manual attention" not in out

    def test_diff_does_not_write_in_place(
        self, sample_md: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        before = sample_md.read_text(encoding="utf-8")
        main(["fix", str(sample_md), "--diff"])
        assert sample_md.read_text(encoding="utf-8") == before

    def test_diff_in_stdin_uses_dash_path(
        self, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import io
        import sys as _sys

        monkeypatch.setattr(_sys, "stdin", io.StringIO("We leverage the data.\n"))
        rc = main(["fix", "-", "--diff"])
        out = capsys.readouterr().out
        assert "--- a/-" in out
        assert "+We use the data." in out
        assert rc == 0

    def test_llm_flag_without_key_reports_error(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        p = tmp_path / "q.md"
        p.write_text(
            "- **Speed:** fast.\n- **Cost:** cheap.\n- **Scale:** easy.\n", encoding="utf-8"
        )
        monkeypatch.delenv("INFERENCE_API_KEY", raising=False)
        rc = main(["fix", str(p), "--llm", "--json"])
        payload = json.loads(capsys.readouterr().out)
        assert isinstance(payload, list) and len(payload) == 1
        assert payload[0]["errors"] == ["INFERENCE_API_KEY is not set"]
        assert rc == 1


class TestCliSurface:
    def test_help(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit) as e:
            main(["--help"])
        assert e.value.code == 0
        assert "detect" in capsys.readouterr().out

    def test_subcommand_help(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit) as e:
            main(["fix", "--help"])
        assert e.value.code == 0
        assert "--llm" in capsys.readouterr().out

    def test_version(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit) as e:
            main(["--version"])
        assert e.value.code == 0

    def test_missing_command_errors(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit) as e:
            main([])
        assert e.value.code != 0

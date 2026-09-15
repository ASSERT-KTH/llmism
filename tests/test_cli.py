from __future__ import annotations

import json
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
    def test_table_output(self, sample_md: Path, capsys: pytest.CaptureFixture[str]) -> None:
        rc = main(["detect", str(sample_md)])
        out = capsys.readouterr().out
        assert "leverage" in out and "its-not-x-its-y" in out
        assert "lexical" in out
        assert "print" not in out  # code fence skipped
        assert rc == 1

    def test_json_output(self, sample_md: Path, capsys: pytest.CaptureFixture[str]) -> None:
        main(["detect", str(sample_md), "--json"])
        payload = json.loads(capsys.readouterr().out)
        assert any(f["rule"] == "leverage-verb" for f in payload)
        assert all("start" in f and "category" in f for f in payload)

    def test_stdin(
        self, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import io

        monkeypatch.setattr("sys.stdin", io.StringIO("We leverage."))
        main(["detect", "-"])
        assert "leverage" in capsys.readouterr().out

    def test_clean_file_returns_zero(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        p = tmp_path / "ok.txt"
        p.write_text("Two bugs were found. Both fixed.", encoding="utf-8")
        rc = main(["detect", str(p)])
        assert rc == 0
        assert "no LLMisms" in capsys.readouterr().out

    def test_format_override_for_stdin(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        import io

        monkeypatch.setattr("sys.stdin", io.StringIO("a `leverage` b"))
        rc = main(["detect", "-", "--format", "markdown"])
        assert rc == 0
        assert "no LLMisms" in capsys.readouterr().out

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
    def test_stdout_rewrite(self, sample_md: Path, capsys: pytest.CaptureFixture[str]) -> None:
        rc = main(["fix", str(sample_md)])
        out = capsys.readouterr().out
        assert "We use the data" in out  # prose rewritten
        assert "print('leverage')" in out  # code fence untouched
        assert "It's not a bug" in out  # flag-only pattern kept
        assert rc == 1  # remaining manual findings

    def test_json_fix(self, sample_md: Path, capsys: pytest.CaptureFixture[str]) -> None:
        main(["fix", str(sample_md), "--json"])
        payload = json.loads(capsys.readouterr().out)
        assert any(f["rule"] == "leverage-verb" for f in payload["fixed"])
        assert any(f["rule"] == "its-not-x-its-y" for f in payload["remaining"])
        assert "We use the data" in payload["text"]

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

    def test_llm_flag_without_key_reports_error(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        p = tmp_path / "q.md"
        p.write_text(
            "- **Speed:** fast.\n- **Cost:** cheap.\n- **Scale:** easy.\n", encoding="utf-8"
        )
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        rc = main(["fix", str(p), "--llm", "--json"])
        payload = json.loads(capsys.readouterr().out)
        assert payload["errors"] == ["ANTHROPIC_API_KEY is not set"]
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

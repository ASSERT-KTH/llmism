from __future__ import annotations

from pathlib import Path

import pytest

from llmism._targets import expand_targets
from llmism.cli import main


@pytest.fixture()
def docs(tmp_path: Path) -> Path:
    (tmp_path / "docs" / "guides").mkdir(parents=True)
    (tmp_path / "docs" / "index.md").write_text("We leverage the data.\n", encoding="utf-8")
    (tmp_path / "docs" / "guides" / "api.md").write_text(
        "It's not a bug, it's a feature.\n", encoding="utf-8"
    )
    (tmp_path / "docs" / "guides" / "clean.txt").write_text("Plain and fine.\n", encoding="utf-8")
    (tmp_path / "docs" / "img.bin").write_bytes(b"\x00")
    (tmp_path / "docs" / ".git").mkdir()
    (tmp_path / "docs" / ".git" / "hidden.md").write_text(
        "We leverage this too.\n", encoding="utf-8"
    )
    return tmp_path / "docs"


class TestExpandTargets:
    def test_file_passes_through(self, docs: Path) -> None:
        assert expand_targets(str(docs / "index.md")) == [str(docs / "index.md")]

    def test_stdin_passes_through(self) -> None:
        assert expand_targets("-") == ["-"]

    def test_directory_walk(self, docs: Path) -> None:
        files = expand_targets(str(docs))
        names = [Path(f).name for f in files]
        assert names == ["api.md", "clean.txt", "index.md"]  # sorted, .bin skipped

    def test_vcs_dirs_skipped(self, docs: Path) -> None:
        assert not any(".git" in f for f in expand_targets(str(docs)))

    def test_format_narrows_walk(self, docs: Path) -> None:
        only_tex = expand_targets(str(docs), "latex")
        assert only_tex == []
        only_txt = expand_targets(str(docs), "text")
        assert [Path(f).name for f in only_txt] == ["clean.txt"]

    def test_missing_path_passes_through(self) -> None:
        assert expand_targets("nope.md") == ["nope.md"]

    def test_include_ext_from_config(self, docs: Path) -> None:
        from llmism._config import Config

        (docs / "notes.rst").write_text("We leverage the data.\n", encoding="utf-8")
        cfg = Config(include=frozenset({".rst"}))
        names = [Path(f).name for f in expand_targets(str(docs), config=cfg)]
        assert names == ["api.md", "clean.txt", "index.md", "notes.rst"]

    def test_exclude_dir_from_config(self, docs: Path) -> None:
        from llmism._config import Config

        cfg = Config(exclude=frozenset({"guides"}))
        names = [Path(f).name for f in expand_targets(str(docs), config=cfg)]
        assert names == ["index.md"]


class TestDetectDirectory:
    def test_json_per_file(self, docs: Path, capsys: pytest.CaptureFixture[str]) -> None:
        import json

        rc = main(["detect", str(docs)])
        payload = json.loads(capsys.readouterr().out)
        paths = {f["path"] for f in payload}
        assert paths == {str(docs / "index.md"), str(docs / "guides" / "api.md")}
        assert rc == 1

    def test_human_summary(self, docs: Path, capsys: pytest.CaptureFixture[str]) -> None:
        rc = main(["detect", str(docs), "--human"])
        out = capsys.readouterr().out
        assert "2 finding(s) in 3 file(s)" in out
        assert "leverage-verb" in out
        assert rc == 1

    def test_clean_tree_exits_zero(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        import json

        (tmp_path / "docs").mkdir()
        (tmp_path / "docs" / "ok.md").write_text("Two bugs found.\n", encoding="utf-8")
        rc = main(["detect", str(tmp_path / "docs")])
        assert json.loads(capsys.readouterr().out) == []
        assert rc == 0

    def test_empty_dir(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        import json

        (tmp_path / "empty").mkdir()
        rc = main(["detect", str(tmp_path / "empty")])
        assert json.loads(capsys.readouterr().out) == []
        assert rc == 0


class TestFixDirectory:
    def test_requires_in_place_or_diff(
        self, docs: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        rc = main(["fix", str(docs)])
        assert rc == 2
        assert "--in-place" in capsys.readouterr().err

    def test_diff_concatenates(self, docs: Path, capsys: pytest.CaptureFixture[str]) -> None:
        rc = main(["fix", str(docs), "--diff"])
        out = capsys.readouterr().out
        assert f"--- a/{docs / 'index.md'}" in out
        assert "+We use the data." in out
        assert rc == 1  # its-not-x-its-y remains

    def test_in_place_rewrites_tree(self, docs: Path, capsys: pytest.CaptureFixture[str]) -> None:
        rc = main(["fix", str(docs), "--in-place", "--human"])
        assert "We use the data." in (docs / "index.md").read_text(encoding="utf-8")
        assert "leverage" in (docs / ".git" / "hidden.md").read_text(encoding="utf-8")
        assert rc == 1

    def test_json_multi_file(self, docs: Path, capsys: pytest.CaptureFixture[str]) -> None:
        rc = main(["fix", str(docs), "--in-place"])
        out = capsys.readouterr().out
        assert out.startswith("[") and out.rstrip().endswith("]")  # one JSON array
        import json

        payloads = json.loads(out)
        assert isinstance(payloads, list) and len(payloads) == 3  # three scannable files
        assert rc == 1

    def test_single_file_still_works(self, docs: Path, capsys: pytest.CaptureFixture[str]) -> None:
        rc = main(["fix", str(docs / "index.md"), "--in-place"])
        assert "We use the data." in (docs / "index.md").read_text(encoding="utf-8")
        assert rc == 1  # the input contained a problem, even though it was fixed

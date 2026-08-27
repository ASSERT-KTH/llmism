"""Command-line interface: ``llmism detect`` and ``llmism fix``."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from .detector import Detector, Finding
from .remediator import Remediator

__all__ = ["main", "build_parser"]

_FORMATS = ("text", "markdown", "latex")
_EXT_FORMATS = {".md": "markdown", ".markdown": "markdown", ".tex": "latex", ".txt": "text"}


def _guess_format(path: str) -> str:
    if path == "-":
        return "text"
    return _EXT_FORMATS.get(Path(path).suffix.lower(), "text")


def _read(path: str) -> str:
    return sys.stdin.read() if path == "-" else Path(path).read_text(encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llmism",
        description="Detect and fix LLM-sounding idioms in prose.",
    )
    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("path", help="file to process, or '-' for stdin")
        p.add_argument(
            "--format",
            choices=_FORMATS,
            default=None,
            help="input format (default: auto-detect from extension, else text)",
        )
        p.add_argument("--json", action="store_true", help="machine-readable output")

    p_detect = sub.add_parser("detect", help="report LLMisms without changing anything")
    add_common(p_detect)

    p_fix = sub.add_parser("fix", help="rewrite LLMisms where a safe fix exists")
    add_common(p_fix)
    p_fix.add_argument("--llm", action="store_true", help="use an LLM for structural fixes")
    p_fix.add_argument("--in-place", action="store_true", help="rewrite the input file")
    return parser


def _findings_payload(findings: list[Finding], path: str) -> list[dict[str, object]]:
    return [
        {
            "path": path,
            "category": f.category,
            "rule": f.rule_id,
            "start": f.start,
            "end": f.end,
            "match": f.matched_text,
            "message": f.message,
            "suggestion": f.suggestion,
        }
        for f in findings
    ]


def _line_of(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def _render_table(rows: Sequence[tuple[str, ...]], widths: tuple[int, ...]) -> str:
    def fmt(row: tuple[str, ...]) -> str:
        cells = [
            c[: w - 1] + "…" if len(c) > w else c.ljust(w)
            for c, w in zip(row, widths, strict=False)
        ]
        return "  ".join(cells).rstrip()

    lines = [fmt(rows[0]), "  ".join("-" * w for w in widths).rstrip()]
    lines += [fmt(r) for r in rows[1:]]
    return "\n".join(lines)


def _print_findings(findings: list[Finding], text: str, path: str) -> None:
    if not findings:
        print(f"{path}: no LLMisms detected")
        return
    rows = [("LINE", "CATEGORY", "RULE", "MATCH")]
    rows += [
        (str(_line_of(text, f.start)), f.category, f.rule_id, repr(f.matched_text[:40]))
        for f in findings
    ]
    print(_render_table(rows, (6, 10, 24, 42)))
    print(f"\n{len(findings)} finding(s)")


def cmd_detect(args: argparse.Namespace) -> int:
    text = _read(args.path)
    fmt = args.format or _guess_format(args.path)
    findings = Detector().scan(text, fmt)
    if args.json:
        print(json.dumps(_findings_payload(findings, args.path), indent=2))
    else:
        _print_findings(findings, text, args.path)
    return 1 if findings else 0


def cmd_fix(args: argparse.Namespace) -> int:
    text = _read(args.path)
    fmt = args.format or _guess_format(args.path)
    findings = Detector().scan(text, fmt)
    result = Remediator().fix(text, findings, use_llm=args.llm)

    if args.json:
        payload: dict[str, object] = {
            "path": args.path,
            "text": result.text,
            "fixed": _findings_payload(result.fixed, args.path),
            "llm_fixed": _findings_payload(result.llm_fixed, args.path),
            "remaining": _findings_payload(result.remaining, args.path),
            "errors": result.errors,
        }
        print(json.dumps(payload, indent=2))
    else:
        for e in result.errors:
            print(f"error: {e}", file=sys.stderr)
        auto = len(result.fixed) + len(result.llm_fixed)
        print(f"{args.path}: {auto} fixed, {len(result.remaining)} need manual attention")
        for f in result.remaining:
            print(f"  - [{f.rule_id}] {f.message}")
        if result.remaining:
            print("  (rerun with --llm to rewrite these automatically)")
    if args.in_place and args.path != "-":
        Path(args.path).write_text(result.text, encoding="utf-8")
    elif not args.json:
        sys.stdout.write(result.text)
        if not result.text.endswith("\n"):
            sys.stdout.write("\n")
    return 1 if result.remaining else 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "detect":
        return cmd_detect(args)
    if args.command == "fix":
        return cmd_fix(args)
    parser.error(f"unknown command {args.command!r}")  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

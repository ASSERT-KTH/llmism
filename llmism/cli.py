"""Command-line interface: ``llmism detect`` and ``llmism fix``."""

from __future__ import annotations

import argparse
import difflib
import json
import shlex
import sys
from pathlib import Path

from ._config import Config, ConfigError, load_config
from ._targets import expand_targets
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
        p.add_argument("path", help="file, directory, or '-' for stdin")
        p.add_argument(
            "--format",
            choices=_FORMATS,
            default=None,
            help="input format (default: auto-detect from extension, else text)",
        )
        p.add_argument(
            "--config",
            default=".llmism.toml",
            help="per-rule config file (default: .llmism.toml in the cwd)",
        )
        p.add_argument(
            "--human",
            dest="json",
            action="store_false",
            help="human-readable output (JSON is the default)",
        )
        p.add_argument(
            "--json",
            dest="json",
            action="store_true",
            help="machine-readable output (default; accepted for compatibility)",
        )

    p_detect = sub.add_parser("detect", help="report LLMisms without changing anything")
    add_common(p_detect)

    p_fix = sub.add_parser("fix", help="rewrite LLMisms where a safe fix exists")
    add_common(p_fix)
    p_fix.add_argument("--llm", action="store_true", help="use an LLM for structural fixes")
    p_fix.add_argument("--in-place", action="store_true", help="rewrite the input file")
    p_fix.add_argument(
        "--diff",
        action="store_true",
        help="print a unified diff of the rewrite instead of the text (review before applying)",
    )
    return parser


def _fix_payload(f: Finding, path: str) -> dict[str, str]:
    """How to fix one finding: deterministic replacement or LLM rewrite."""
    if f.suggestion is not None:
        return {
            "kind": "replace",
            "replacement": f.suggestion,
            "command": f"llmism fix {shlex.quote(path)}",
        }
    return {
        "kind": "llm",
        "command": f"llmism fix {shlex.quote(path)} --llm",
        "hint": f.message,
    }


def _findings_payload(findings: list[Finding], text: str, path: str) -> list[dict[str, object]]:
    return [
        {
            "path": path,
            "line": _line_of(text, f.start),
            "category": f.category,
            "rule": f.rule_id,
            "start": f.start,
            "end": f.end,
            "match": f.matched_text,
            "message": f.message,
            "suggestion": f.suggestion,
            "fix": _fix_payload(f, path),
        }
        for f in findings
    ]


def _line_of(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def _diagnostic(f: Finding, text: str, path: str) -> str:
    # Compiler-style diagnostic with a one-indexed source line.
    return f"{path}:{_line_of(text, f.start)}: error: [{f.category}/{f.rule_id}] {f.message}"


def _print_findings(findings: list[Finding], text: str, path: str) -> None:
    for finding in findings:
        print(_diagnostic(finding, text, path))
    print(f"\n{len(findings)} finding(s)")


def _load_config_or_exit(path: str) -> Config:
    try:
        return load_config(path)
    except ConfigError as e:
        print(f"error: {e}", file=sys.stderr)
        raise SystemExit(2) from e


def cmd_detect(args: argparse.Namespace) -> int:
    config = _load_config_or_exit(args.config)
    targets = expand_targets(args.path, args.format, config)
    detector = Detector(config=config)
    all_findings: list[tuple[str, str, list[Finding]]] = []
    for target in targets:
        text = _read(target)
        fmt = args.format or _guess_format(target)
        all_findings.append((target, text, detector.scan(text, fmt)))

    if args.json:
        payload: list[dict[str, object]] = []
        for target, text, findings in all_findings:
            payload.extend(_findings_payload(findings, text, target))
        print(json.dumps(payload, indent=2))
    else:
        total = 0
        for target, text, findings in all_findings:
            if findings:
                _print_findings(findings, text, target)
                total += len(findings)
        if len(targets) > 1:
            print(f"\n{total} finding(s) in {len(targets)} file(s)")
        elif not total:
            print(f"{targets[0] if targets else args.path}: no LLMisms detected")
    return 1 if any(f for _, _, f in all_findings) else 0


def _unified_diff(original: str, rewritten: str, path: str) -> str:
    """Git-style unified diff of the rewrite (``a/<path>`` vs ``b/<path>``)."""
    return "".join(
        difflib.unified_diff(
            original.splitlines(keepends=True),
            rewritten.splitlines(keepends=True),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
        )
    )


def cmd_fix(args: argparse.Namespace) -> int:
    config = _load_config_or_exit(args.config)
    if Path(args.path).is_dir() and not (args.in_place or args.diff):
        print(
            "error: --in-place (or --diff to review) is required when fixing a directory",
            file=sys.stderr,
        )
        return 2
    targets = expand_targets(args.path, args.format, config)
    detector = Detector(config=config)
    if args.json and not args.diff:
        # collect every payload first, then print one JSON array
        payloads: list[dict[str, object]] = []
        rc = 0
        for target in targets:
            payload, target_rc = _fix_file_payload(args, target, detector)
            payloads.append(payload)
            rc = max(rc, target_rc)
        print(json.dumps(payloads, indent=2))
        return rc
    rc = 0
    for target in targets:
        rc = max(rc, _fix_stream(args, target, detector))
    return rc


def _fix_file_payload(
    args: argparse.Namespace, target: str, detector: Detector
) -> tuple[dict[str, object], int]:
    """Run fix on one target and return (JSON payload, exit code)."""
    text = _read(target)
    fmt = args.format or _guess_format(target)
    findings = detector.scan(text, fmt)
    result = Remediator().fix(text, findings, use_llm=args.llm)
    for e in result.errors:
        print(f"error: {e}", file=sys.stderr)
    if args.in_place and target != "-":
        Path(target).write_text(result.text, encoding="utf-8")
    payload: dict[str, object] = {
        "path": target,
        "text": result.text,
        "diff": _unified_diff(text, result.text, target),
        "fixed": _findings_payload(result.fixed, text, target),
        "llm_fixed": _findings_payload(result.llm_fixed, text, target),
        "remaining": _findings_payload(result.remaining, text, target),
        "errors": result.errors,
    }
    return payload, (1 if result.remaining or result.errors else 0)


def _fix_stream(args: argparse.Namespace, target: str, detector: Detector) -> int:
    """Run fix on one target, streaming human/diff output. Returns exit code."""
    text = _read(target)
    fmt = args.format or _guess_format(target)
    findings = detector.scan(text, fmt)
    result = Remediator().fix(text, findings, use_llm=args.llm)
    for e in result.errors:
        print(f"error: {e}", file=sys.stderr)

    if args.diff:
        diff = _unified_diff(text, result.text, target)
        if diff:
            sys.stdout.write(diff)
            if not diff.endswith("\n"):
                sys.stdout.write("\n")
    else:
        if not findings:
            pass
        elif not result.remaining:
            print("llmism fixed")
        else:
            if args.in_place:
                print("llmism fixed")
            print(f"{target}: {len(result.remaining)} need manual attention")
            for f in result.remaining:
                print(_diagnostic(f, text, target))
        if result.remaining:
            print("  (rerun with --llm to rewrite these automatically)")
        if not args.in_place:
            sys.stdout.write(result.text)
            if not result.text.endswith("\n"):
                sys.stdout.write("\n")
    if args.in_place and target != "-":
        Path(target).write_text(result.text, encoding="utf-8")
    return 1 if result.remaining or result.errors else 0


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

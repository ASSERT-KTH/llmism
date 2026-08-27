"""Lazy-imported Anthropic wrapper used only when ``--llm`` is passed."""

from __future__ import annotations

import os

__all__ = ["rewrite_spans", "LLMError"]

_SYSTEM = (
    "You are a precise copy editor. Rewrite the given text spans so they no longer "
    "sound machine-generated, preserving meaning, facts, Markdown/LaTeX structure and "
    "language. Return only a JSON object mapping each span id to its rewrite."
)


class LLMError(RuntimeError):
    """Raised when the LLM rewrite path fails (missing key, API error, bad output)."""


def rewrite_spans(
    text: str,
    spans: list[tuple[int, int, str]],
    *,
    model: str = "claude-sonnet-4-5",
    max_chars_per_span: int = 1200,
) -> dict[int, str]:
    """Rewrite ``[(start, end, matched_text), ...]`` spans of ``text``.

    Returns ``{span_index: replacement}`` (index into the input list).
    """
    try:
        import anthropic  # noqa: PLC0415 - deliberately lazy
    except ImportError as e:  # pragma: no cover - exercised via stub tests
        msg = "the 'llm' extra is required for --llm: pip install llmism[llm]"
        raise LLMError(msg) from e

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        msg = "ANTHROPIC_API_KEY is not set"
        raise LLMError(msg)

    client = anthropic.Anthropic(api_key=api_key)
    payload: dict[str, dict[str, str]] = {}
    for i, (_start, _end, matched) in enumerate(spans):
        snippet = matched[:max_chars_per_span]
        context_before = text[max(0, _start - 200) : _start].replace("\n", " ")
        context_after = text[_end : _end + 200].replace("\n", " ")
        payload[str(i)] = {
            "text": snippet,
            "context_before": context_before[-200:],
            "context_after": context_after[:200],
        }

    import json

    response = client.messages.create(
        model=model,
        max_tokens=2048,
        system=_SYSTEM,
        messages=[{"role": "user", "content": "Rewrite these spans:\n" + json.dumps(payload)}],
    )
    raw = "".join(block.text for block in response.content if block.type == "text")
    try:
        parsed = json.loads(raw)
        return {int(k): str(v) for k, v in parsed.items()}
    except (ValueError, TypeError) as e:
        msg = f"LLM returned non-JSON output: {raw[:200]!r}"
        raise LLMError(msg) from e

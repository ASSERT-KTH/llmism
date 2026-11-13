"""Lazy-imported Chat Completions wrapper used only when ``--llm`` is passed.

Speaks the OpenAI-compatible Chat Completions API against any conforming
server. Configure via environment:

- ``INFERENCE_API_KEY`` (required) — bearer key for the server
- ``INFERENCE_BASE_URL`` (optional) — server root, e.g. ``http://localhost:8000/v1``
- ``INFERENCE_MODEL`` (optional) — model name, default ``gpt-4o-mini``
"""

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
    model: str | None = None,
    max_chars_per_span: int = 1200,
) -> dict[int, str]:
    """Rewrite ``[(start, end, matched_text), ...]`` spans of ``text``.

    Returns ``{span_index: replacement}`` (index into the input list).
    """
    api_key = os.environ.get("INFERENCE_API_KEY")
    if not api_key:
        msg = "INFERENCE_API_KEY is not set"
        raise LLMError(msg)

    try:
        from openai import OpenAI  # noqa: PLC0415 - deliberately lazy
    except ImportError as e:  # pragma: no cover - exercised via stub tests
        msg = "the 'llm' extra is required for --llm: pip install llmism[llm]"
        raise LLMError(msg) from e

    client = OpenAI(api_key=api_key, base_url=os.environ.get("INFERENCE_BASE_URL"))
    resolved_model = model or os.environ.get("INFERENCE_MODEL") or "gpt-4o-mini"

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

    response = client.chat.completions.create(
        model=resolved_model,
        max_tokens=2048,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": "Rewrite these spans:\n" + json.dumps(payload)},
        ],
    )
    raw = response.choices[0].message.content or ""
    try:
        parsed = json.loads(raw)
        return {int(k): str(v) for k, v in parsed.items()}
    except (ValueError, TypeError) as e:
        msg = f"LLM returned non-JSON output: {raw[:200]!r}"
        raise LLMError(msg) from e

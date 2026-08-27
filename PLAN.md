# LLMism detector + remediator (local prototype)

## Context

`README.md` asks for a plan to detect "LLMisms" — idioms/tics overused by LLMs (em-dash abuse, "it's not X, it's Y", "delve", bolded-list-item spam, uniform sentence rhythm, etc.) — and a remediator that fixes them. Research confirms three tiers of tell:

- **Lexical**: single overused words (delve, tapestry, boundaries, leverage, pivotal, underscores, elevate).
- **Phrasal**: fixed-ish patterns ("it's not X, it's Y", "not only X but also Y", "while X has benefits, it also carries risks", self-posed rhetorical Q&A).
- **Structural/stylometric**: em-dash overuse, every bullet starting with a bolded lead-in, low sentence-length burstiness, formulaic transition-word density (furthermore/moreover/in addition).

Decisions:
- **Remediation**: hybrid — deterministic rule-based rewrites by default (offline, testable); optional `--llm` flag calls an LLM for structural cases rules can't safely rewrite.
- **Scan targets**: plain text, Markdown, and LaTeX (need to strip code fences / math environments from scanning, and markdown needs bullet/bold-lead-in detection).
- **Packaging**: local prototype only — skip CI/PyPI publishing setup for now, but still follow standard Python conventions (flat layout, hatchling, pytest, mypy+ruff) so it's a one-step upgrade later.

## Package layout (flat, per convention)

```
llmism/
  __init__.py
  _patterns.py       # lexical + phrasal pattern tables (data structures, not the raw data)
  _markup.py         # format-aware helpers: strip fenced code / LaTeX math envs, detect bullet+bold lead-ins, sentence splitter
  detector.py         # Detector class + Finding dataclass
  remediator.py        # Remediator class: rule-based fixups; optional LLM rewrite path
  _llm.py           # lazy-imported Anthropic client wrapper, only touched when --llm used
  cli.py            # argparse entry point
  data/
    patterns.yaml       # bundled seed list of lexical/phrasal LLMisms with category + suggested replacement
tests/
  test_detector.py
  test_remediator.py
  test_markup.py
  test_cli.py
pyproject.toml         # hatchling backend, [project.scripts] llmism = "llmism.cli:main"
README.md            # update with usage once implemented
LICENSE             # MIT
```

## Core design

**`Finding` (dataclass)**: `category` (`lexical`|`phrasal`|`structural`), `rule_id`, `start`/`end` offsets, `matched_text`, `message`, `suggestion` (`str | None` — present only when a safe rule-based rewrite exists).

**`Detector.scan(text, fmt="markdown") -> list[Finding]`**
- `_markup.py` first computes "scannable ranges" for the given format: for markdown, exclude fenced code blocks and inline code; for latex, exclude `\begin{verbatim}`/math environments; for plain text, whole doc.
- Lexical/phrasal rules run as regex/word-boundary scans over `data/patterns.yaml` entries (case-insensitive, word-boundary aware).
- Structural rules are small dedicated functions:
  - em-dash density (per N words) above threshold.
  - markdown-only: consecutive bullet items where >X% start with `**bold**`.
  - rhetorical-question-immediately-answered (regex: sentence ending in `?` followed by a sentence starting "The answer is/Yes/No/It is").
  - burstiness: stdev of sentence lengths below threshold relative to mean (flag whole doc, not a span).

**`Remediator.fix(text, findings, use_llm=False) -> str`**
- For `lexical`/`phrasal` findings with a `suggestion` in patterns.yaml: apply as direct substring replacement (sorted by offset, applied right-to-left to keep offsets valid).
- For `structural` findings (no safe deterministic rewrite): if `use_llm`, batch them with surrounding context into one Anthropic API call (`_llm.py`, lazy import so the package works with zero extra deps offline) asking for a meaning-preserving rewrite of just those spans; otherwise leave text untouched and surface them as remaining findings for the user to fix by hand.
- Returns fixed text; CLI reports what was auto-fixed vs. what still needs manual/`--llm` attention.

**CLI (`llmism/cli.py`)**
- `llmism detect <path|-> [--format text|markdown|latex] [--json]`
- `llmism fix <path|-> [--format ...] [--llm] [--in-place] [--json]`
- Format auto-detected from file extension when not passed (`.md`→markdown, `.tex`→latex, else text).
- Clean console table output by default (per convention); `--json` for machine-readable findings.

## Seed pattern data

`data/patterns.yaml` seeded from research (McGill OSS article, tropes.fyi gist, stylometric detection papers) — entries like:

```yaml
- id: delve
  category: lexical
  pattern: '\bdelve(?:s|d|ing)?\b'
  suggestion: "explore / look into"
- id: its-not-x-its-y
  category: phrasal
  pattern: "it'?s not [^,.;]+,? it'?s "
  suggestion: null   # structural-ish, flag only
- id: boundaries
  category: lexical
  pattern: '\bboundaries\b'
  suggestion: "limits"
...
```
Keep this list small and curated at first (~20-30 entries across lexical/phrasal), expandable later without code changes.

## Dev tooling (still per convention, minus CI/publish)

- `pyproject.toml`: hatchling build backend, `[project.scripts] llmism = "llmism.cli:main"`, dev deps: pytest, pytest-cov, mypy, ruff.
- `from __future__ import annotations` + type hints throughout.
- `.gitignore` for `__pycache__`, `.pytest_cache`, `*.egg-info`, etc.
- MIT `LICENSE`.
- No GitHub Actions workflow, no PyPI trusted-publisher setup — deferred until the approach is validated.

## Verification

- `pytest -q` covering: each seed pattern fires on a crafted example and not on a clean paragraph; markdown code-fence content is never flagged; latex math environments are never flagged; `Remediator.fix` produces expected output for lexical/phrasal cases; `--llm` path is exercised with a mocked Anthropic client (no live API calls in tests).
- Manual smoke test: `python -m llmism.cli detect README.md` and `... fix` on a hand-written sample doc containing several deliberate LLMisms, eyeball the findings table and the fixed output.
- `ruff check` and `mypy` clean.

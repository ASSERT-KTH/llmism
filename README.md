# llmism

Detector and remediator for *LLMisms* — idioms and tics overused by language
models that make prose sound machine-generated: `delve`, "it's not X, it's Y",
em-dash overuse, bolded bullet lead-ins, etc.

## Install (local prototype)

```sh
pip install -e ".[dev]"
```

## Detect

```sh
llmism detect README.md          # console table, exit 1 if findings
llmism detect doc.tex            # .tex → LaTeX, .md → Markdown, else text
llmism detect - --format markdown < draft.txt
llmism detect post.md --json     # machine-readable findings
```

```
LINE  CATEGORY   RULE                     MATCH
-----  ---------  -----------------------  ------------------------------------
3      lexical    delve                    'delve'
3      lexical    leverage-verb            'leverage'
5      phrasal    its-not-x-its-y          "It's not a bug, it's "
```

## Fix

```sh
llmism fix post.md                    # rewritten text on stdout
llmism fix post.md --in-place         # rewrite the file
llmism fix post.md --json             # {text, fixed, remaining, ...}
```

- **Lexical/phrasal findings with a safe replacement** are rewritten
  deterministically (capitalisation preserved, offsets applied right-to-left).
- **Structural findings** (em-dash density, bold-lead-in bullets, colon-hinged
  sentences, low burstiness) have no safe deterministic rewrite. They are
  reported as `remaining` unless you pass `--llm`:

```sh
pip install -e ".[llm]"   # optional anthropic extra
export ANTHROPIC_API_KEY=...
llmism fix post.md --llm --in-place
```

## What is detected

| Tier | Rules |
|---|---|
| lexical | `genuine`, `merely`, `leverage`, `underscores`, `elevate`, `robust`, `load-bearing`, `boundaries`, `honestly`, `sufficient`, `terminate`, `approximately` |
| phrasal | "it's not X, it's Y", `rather than`, `in addition`, `-ing` tail clauses, colon reveals ("here's the thing"), depth-signalling ("at a more fundamental level"), announcing labels ("the key insight is"), engagement bait ("let me know if") |
| structural | colon-hinged sentences, ≥3 bullets with `**bold**` lead-ins, verbless fragments used as sentences, headers written as sentences, em-dash overuse (per paragraph), sentence-opener repetition, hedge stacking, uniform sentence rhythm (low burstiness), uniform paragraph sizes, mechanical short/long cadence, synonym cycling, degenerate repetition |

Further rules flag chatbot openings, vague attributions, generic conclusions,
inflated significance, unfilled placeholders, leaked citation markup, AI-tool URL
parameters, and lines with six or more social hashtags. `in order to` and
`due to the fact that` have deterministic fixes; claims and tool artifacts stay
for review.

The rule set is pruned to what actually fires: every rule here hit at least 10
times over ~15k assistant turns (1.7M words) of Claude Code sessions. Rules that
never paid for their scan time (`delve`, `tapestry`, `furthermore`, `utilize`,
self-answered rhetorical questions, transition clusters, ...) live in git history.

Pattern data lives in `llmism/data/patterns.yaml` — extend the list without
touching code. Code fences (Markdown) and math environments, verbatim blocks and
`%` comments (LaTeX) are never scanned or rewritten. Several structural rules are
inspired by the [sloptrim](https://github.com/seyedehsanhadi/sloptrim) pattern
catalogue; the colon, fragment, header, depth-signalling and Anglo-Saxon-over-Latinate
rules come from the
[claude-style-patch](https://github.com/andrewroxby/claude-style-patch) house-style
guide. The newer artifact, attribution, conclusion, and hashtag rules draw on
the [avoid-ai-writing](https://github.com/conorbronsdon/avoid-ai-writing)
pattern catalog.

## Library use

```python
from llmism import Detector, Remediator

findings = Detector().scan(text, "markdown")
result = Remediator().fix(text, findings)
print(result.remaining)   # findings left for manual fixing
```

## Development

```sh
pytest -q --cov=llmism
ruff check . && mypy llmism
```

MIT licensed. See [LICENSE](LICENSE).

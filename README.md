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
- **Structural findings** (em-dash density, bold-lead-in bullets, rhetorical
  Q&A pairs, low burstiness) have no safe deterministic rewrite. They are
  reported as `remaining` unless you pass `--llm`:

```sh
pip install -e ".[llm]"   # optional anthropic extra
export ANTHROPIC_API_KEY=...
llmism fix post.md --llm --in-place
```

## What is detected

| Tier | Rules |
|---|---|
| lexical | `delve`, `tapestry`, `pivotal`, `leverage`, `seamless`, `boundaries`, `robust`, `load-bearing`, ... |
| phrasal | "it's not X, it's Y", "not only X but also Y", `-ing` tail clauses, vague attributions ("studies suggest"), cataloguing lead-ins ("uses several mechanisms... These methods..."), empty pivots ("it's worth noting"), significance inflation ("stands as a testament to"), `Furthermore`/`Moreover`, ... |
| structural | em-dash overuse (per paragraph), transition-word clusters, sentence-opener repetition, hedge stacking, ≥3 bullets with `**bold**` lead-ins, self-answered rhetorical questions, uniform sentence rhythm (low burstiness), uniform paragraph sizes, mechanical short/long cadence, synonym cycling, degenerate repetition |

Pattern data lives in `llmism/data/patterns.yaml` — extend the list without
touching code. Code fences (Markdown) and math environments, verbatim blocks and
`%` comments (LaTeX) are never scanned or rewritten. Several structural rules are
inspired by the [sloptrim](https://github.com/seyedehsanhadi/sloptrim) pattern
catalogue.

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

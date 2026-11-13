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
llmism detect README.md          # JSON findings on stdout, exit 1 if any
llmism detect doc.tex            # .tex → LaTeX, .md → Markdown, else text
llmism detect - --format markdown < draft.txt
llmism detect post.md --human    # console table for humans
```

### Directory input

`path` can be a directory; the tree is walked recursively (`.git`,
`node_modules`, build dirs, ... are skipped) for scannable extensions
(`.md`, `.markdown`, `.tex`, `.txt`), with per-file reporting:

```sh
llmism detect docs/              # JSON array across all files, exit 1 if any
llmism detect docs/ --human      # one table per file + a summary line
llmism detect . --format markdown   # restrict the walk to .md/.markdown
```

JSON output is a flat findings list where every entry carries its own
`path`; `fix --diff` concatenates one unified diff per changed file, so
`llmism fix docs/ --diff | git apply -` works on the whole tree.

### Per-rule configuration

A `.llmism.toml` in the working directory (override with `--config`)
disables or tunes rules, ruff/stylelint-style:

```toml
# .llmism.toml
ignore = ["low-burstiness", "merely"]      # disable rules everywhere

include = [".rst"]                # extra extensions scanned in directory walks
exclude = ["vendor", "generated"] # extra directory names skipped in walks

[rules.em-dash-overuse]
max-per-paragraph = 4                       # tune a threshold

[rules.leverage-verb]
suggestion = "draw on"                      # override a replacement
```

Structural rules expose their thresholds (`min-cv`, `max-words`, ...);
lexical/phrasal rules accept `suggestion`. Unknown rules or keys are a
hard error (exit 2), so typos never silently no-op.

Each finding carries a `fix` object an agent can act on directly:
`{"kind": "replace", "replacement": "use", "command": "llmism fix post.md"}`
for deterministic fixes, and
`{"kind": "llm", "command": "llmism fix post.md --llm", "hint": "..."}`
for structural findings that need a model rewrite.

```
LINE  CATEGORY   RULE                     MATCH
-----  ---------  -----------------------  ------------------------------------
3      lexical    delve                    'delve'
3      lexical    leverage-verb            'leverage'
5      phrasal    its-not-x-its-y          "It's not a bug, it's "
```

## Fix

```sh
llmism fix post.md                    # JSON payload {text, diff, fixed, remaining, ...}
llmism fix post.md --human            # rewritten text on stdout
llmism fix post.md --diff             # unified diff of the rewrite — review it
llmism fix post.md --llm --diff       # diff the LLM's rewrite before accepting
llmism fix post.md --in-place         # rewrite the file
llmism fix docs/ --diff               # whole-tree diff, pipe into git apply
llmism fix docs/ --in-place           # rewrite every scannable file in docs/
```

Fixing a directory requires `--in-place` or `--diff` (dumping several
rewritten files to stdout would be useless). JSON `fix` output is a
**JSON array** with one payload object per file, for single files and
trees alike.

- **Lexical/phrasal findings with a safe replacement** are rewritten
  deterministically (capitalisation preserved, offsets applied right-to-left).
- **Emoji** are deleted, together with the blank that separated them from the
  text (`Done ✅ and` → `Done and`, `- 🚀 Deploy` → `- Deploy`).
- **Other structural findings** (em-dash density, bold-lead-in bullets, colon-hinged
  sentences, low burstiness) have no safe deterministic rewrite. They are
  reported as `remaining` unless you pass `--llm`:

```sh
pip install -e ".[llm]"   # optional inference extra
export INFERENCE_API_KEY=...            # required
export INFERENCE_BASE_URL=http://...    # optional, Chat Completions server root
export INFERENCE_MODEL=...              # optional, default gpt-4o-mini
llmism fix post.md --llm --in-place
```

The LLM path speaks the OpenAI-compatible Chat Completions API, so any
conforming server works (vLLM, llama.cpp, Ollama, OpenRouter, ...).

## What is detected

| Tier | Rules |
|---|---|
| lexical | `genuine`, `merely`, `leverage`, `underscores`, `elevate`, `robust`, `load-bearing`, `boundaries`, `honestly`, `sufficient`, `terminate`, `approximately` |
| phrasal | "it's not X, it's Y", `rather than`, `in addition`, `-ing` tail clauses, colon reveals ("here's the thing"), depth-signalling ("at a more fundamental level"), announcing labels ("the key insight is"), engagement bait ("let me know if") |
| structural | colon-hinged sentences, ≥3 bullets with `**bold**` lead-ins, any bullet whose bold lead-in is a whole sentence (`- **No prioritisation.**`), emoji in prose, verbless fragments used as sentences, headers written as sentences, em-dash overuse (per paragraph), sentence-opener repetition, hedge stacking, uniform sentence rhythm (low burstiness), uniform paragraph sizes, mechanical short/long cadence, synonym cycling, degenerate repetition |

Further rules flag chatbot openings, vague attributions, generic conclusions,
inflated significance, unfilled placeholders, leaked citation markup, AI-tool URL
parameters, and lines with six or more social hashtags. `in order to` and
`due to the fact that` have deterministic fixes; claims and tool artifacts stay
for review.

The rule set is pruned to what actually fires: every rule here hit at least 10
times over ~15k assistant turns (1.7M words) of Claude Code sessions. Rules that
never paid for their scan time (`delve`, `tapestry`, `furthermore`, `utilize`,
self-answered rhetorical questions, transition clusters, ...) live in git history.
`bold-sentence-bullet` and `emoji` were added later by hand and have not been
measured against that corpus.

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
print(result.remaining)  # findings left for manual fixing
```

## Development

```sh
pytest -q --cov=llmism
ruff check . && mypy llmism
```

MIT licensed. See [LICENSE](LICENSE).

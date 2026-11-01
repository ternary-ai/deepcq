# deepcq

**Deep Context Query** — intent-aware memory querying for LLM agents.

*Remember when it matters.*

deepcq sits between an agent's memory and its prompt. Instead of stuffing an
entire memory blob into every LLM call, it scores which sections of memory
are actually relevant to the current query and returns only those — cutting
prompt size while keeping recall.

No embeddings, no vector store, no external dependencies. Relevance is
computed from the query's own tokens against text already in memory, using
plain regex and a lightweight stemmer.

## Install

```bash
pip install deepcq
```

For local development from a clone:

```bash
cd deepcq
pip install -e ".[dev]"
```

## Core idea

You declare a `MemorySchema`: a list of section names such as
`["status", "errors", "metrics"]`. deepcq doesn't need to know what those
sections mean — it derives what's relevant from the words in each query
itself ("query-term-first"), rather than from a hand-built keyword table.

```python
from deepcq import DeepCQ, MemorySchema

schema = MemorySchema(sections=["status", "errors", "metrics"])
cq = DeepCQ(schema=schema)

cq.state.set("errors",  "[errors]\n- OOM at 02:15\n")
cq.state.set("metrics", "[metrics]\nCPU: 78%\nMemory: 4.2GB\n")
cq.state.set("status",  "[status]\nAll services running\n")

result = cq.query("any memory or cpu spikes?")
print(result.content)                       # only the metrics section
print(result.sections_included)             # ["metrics"]
print(f"{result.metrics.reduction_pct:.1f}% smaller")
```

### `locate()` — finding relevant chunks in a text blob

For the common case of "which part of this document answers this query"
(no schema, no memory tiers, just text), `locate()` wraps `DeepCQ` into a
one-shot convenience function:

```python
from deepcq import locate

doc = (
    "The authentication service validates tokens on every request.\n\n"
    "Rate limiting is enforced per API key using a sliding window counter.\n\n"
    "The database migration tool applies schema changes in order."
)

for candidate in locate(doc, "how does rate limiting work?"):
    print(candidate.index, candidate.score, candidate.text)
```

`locate()` splits `text` on `chunk_on` (default `"\n\n"`), scores each chunk
against `query`, and returns `Candidate(index, text, score)` objects for
chunks at or above `confidence_threshold`, ordered by position in the
original text. `text` is returned unmodified — no headers, no wrapping — so
it's safe to use directly for verbatim matching against the source document.
Pass `max_candidates` to cap the result to the highest-scoring chunks.

## How it works

### 1. Five memory tiers

`DeepCQ` manages five tiers, each suited to a different memory lifetime:

| Tier | Class | Lifetime | Purpose |
|------|-------|----------|---------|
| T1 | `AppState` | live | Current state of the running application (sections keyed by name) |
| T2 | `SessionCache` | per-session, bounded | Ephemeral key/value scratch space, deduped, LRU-evicted |
| T3 | `History` | sliding window | Last N conversation turns |
| T4 | `LongTermStore` | persistent | Append-only notes, optionally file-backed (JSON) |
| T5 | `ReferenceData` | persistent, lazy | Static config/reference data, optionally file-backed |

Each tier renders itself to text via `as_text()`. `DeepCQ.query()` pulls text
from all five tiers before scoring.

### 2. Tokenizing the query (`tokens.py`)

The query is lowercased, split into words, stripped of stopwords, and each
surviving token is paired with a suffix-stripped stem (`spikes` → `spike`,
`running` → `run`). No external NLP dependency — it's a small suffix-rule
table plus a doubled-consonant cleanup (`running` → `runn` → `run`).

### 3. Scoring sections (`intent.py`)

For each section in the schema, deepcq builds a vocabulary from:
1. The section name itself (tokenized/stemmed).
2. Optional `keywords` you declare for that section in the schema (useful
   when a section name is jargon, e.g. `"db"` → `["database", "postgres"]`).
3. Whatever text is currently stored under that section's header across all
   tiers.

The section's score is the fraction of unique query tokens found in that
vocabulary. Sections scoring at or above `schema.confidence_threshold`
(default `0.3`) are "included". A section that no tier contains scores
`0.0` (`ScoreExplanation.present` is `False`) and is listed in
`sections_absent` — name/keyword matches alone never include it. A query
that matches only such absent sections returns no section content (it asks
about data memory doesn't hold) rather than triggering the include-all
fallback; this is logged and counted as `absent_only`.

Literal anchors from the query (quoted phrases, numbers, percentages,
identifiers, emoji) force a section's score to 1.0 when found in its text
as the exact figure or whole word: `"10"` matches `"top 10"` and the range
`"10-15%"` but not `"100"`, `"2010"`, `"2.10"`, `"10.5"` or `"10-K"`;
`"15%"` doesn't match `"115%"` or `"-15%"` (a leading minus sign is part of
the figure). Matching is textual: `"1000"` doesn't match `"1,000"`. For
number-heavy memory (financial data), pass `anchor_patterns=` without the
number patterns.

Extracted section text is cached on each tier object, keyed by a version
counter that every mutating method bumps — repeated queries against
unchanged (e.g. large, static `notes`/`reference`) tiers skip the regex
search and re-tokenization entirely.

#### Opt-in scoring options (0.3.0)

All of these default to off and reduce exactly to the scoring above.

- **Synonym vocabularies** (`vocabulary.py`) — `MemorySchema(vocabulary=...)`
  expands each section's vocabulary with every synonym cluster whose term
  occurs in it, so `"eps"` matches stored `"earnings per share"` and vice
  versa, without per-section `keywords`:

  ```python
  from deepcq import MemorySchema, SynonymMap, load_pack

  vocab = load_pack("finance") | SynonymMap.from_dict({"arr": ["annual recurring revenue"]})
  schema = MemorySchema(sections=["fundamentals", "news"], vocabulary=vocab)
  ```

- **Phrase-aware matching** (`phrases.py`) — `phrase_sizes=(2,)` adds
  `phrase_boost` (default `0.25`) per stemmed n-gram from the query that
  appears verbatim (adjacent) in a section, so "moving average" beats text
  that merely mentions "moving" and "average" apart. Bonus only, capped at
  1.0 — but it does shift scores, so recalibrate thresholds after enabling.

- **Per-tier weighting** — `tier_weights={"history": 0.5}` (and
  `schema_weight` for name/keyword matches) make a match in stale tiers
  count for less. Each query token counts once, at the best weight among
  the places it was found. `ScoreExplanation.tier_contributions` shows
  which tier supplied how much of the score.

### Calibrating thresholds (`calibration.py`)

Instead of hand-tuning `confidence_threshold`, label a few cases and sweep:

```python
from deepcq import calibrate, LabeledExample

def build():                       # one populated DeepCQ per case
    cq = DeepCQ(schema)
    cq.state.set("metrics", "[metrics]\nCPU: 78%\n")
    return cq

report = calibrate([LabeledExample(build, "cpu load?", {"metrics"}), ...])
print(report.summary())            # threshold | precision | recall | f1
schema.confidence_threshold = report.best.threshold
```

Each example is queried once and its raw scores re-thresholded per sweep
step (micro-averaged precision/recall; best F1, ties → higher threshold).
Pass `min_precision=` to instead get the highest-recall threshold meeting
that precision; if none does, `report.constraint_met` is `False` and
`summary()` prints a warning.

### 4. Extracting content (`schema.py`, `regex_gen.py`)

Sections are delimited in stored text by a header, `"[{name}]"` by default
(configurable via `section_header_format`, e.g. `"## {name}"`). Each section
compiles to a regex that captures from its header up to the next header or
end of string — this is how `MemorySchema.extraction_pattern(section)` pulls
a section's block out of raw tier text. A header ending in a letter or
digit must end there, so `"## SEC"` doesn't match `"## Sector"`.

For the two free-text tiers (`cache`, `history`, which aren't
section-delimited), deepcq instead builds a single regex out of all query
tokens and includes that tier's text only if any token matches
(`build_content_pattern` in `regex_gen.py`).

### 5. Result (`result.py`)

`query()` returns a `QueryResult`:

```python
@dataclass
class QueryResult:
    content: str                      # the filtered, concatenated text
    intent: dict[str, float]          # per-section relevance scores
    sections_included: list[str]
    sections_excluded: list[str]
    content_pattern: str              # the regex used against cache/history
    metrics: OptimizationMetrics      # original_chars, filtered_chars, reduction_pct
    explanations: dict[str, ScoreExplanation]  # matched tokens/anchors/phrases, tier_contributions, present
    sections_absent: list[str]        # in the schema, but in no tier
    fallback_used: bool               # see "Fallback safety"
    fallback_reason: str | None
```

`sections_excluded` lists sections that are present but scored below
threshold.

### Fallback safety

If no section clears the confidence threshold and
`schema.fallback_include_all` is `True` (the default), `query()` returns the
full unfiltered text instead of nothing — this prevents a vague query from
silently discarding memory the agent might actually need.

The fallback is never silent: the result has `fallback_used=True` and a
`fallback_reason`, a warning is logged on the `"deepcq"` logger, and
`deepcq.fallback_counts()` returns per-kind totals (`no_section_scored`,
`no_section_present`, plus `absent_only` — see above) since process start or the last
`deepcq.reset_fallback_counts()`. Check them before trusting a reduction
number.

## Package layout

```
deepcq/
├── pyproject.toml
├── README.md
├── deepcq/                      # the installable package
│   ├── __init__.py              # public API: DeepCQ, MemorySchema, QueryResult, ...
│   ├── core.py                  # DeepCQ orchestrator (query())
│   ├── schema.py                # MemorySchema + extraction regex compilation
│   ├── intent.py                # score_sections() / score_sections_detailed()
│   ├── anchors.py               # literal-anchor extraction
│   ├── decay.py                 # Recency + decay functions
│   ├── vocabulary.py            # SynonymMap, load_pack()
│   ├── vocab_packs/             # built-in synonym packs (finance.json)
│   ├── phrases.py               # extract_phrases() for phrase-aware matching
│   ├── calibration.py           # calibrate() threshold sweep
│   ├── locate.py                # locate() chunk retrieval
│   ├── tokens.py                # tokenize() / stem()
│   ├── regex_gen.py              # query-token regex builder for free-text tiers
│   ├── result.py                # QueryResult / OptimizationMetrics dataclasses
│   ├── memory/
│   │   ├── state.py             # T1 AppState
│   │   ├── session.py           # T2 SessionCache, T3 History
│   │   └── store.py             # T4 LongTermStore, T5 ReferenceData
│   └── integrations/
│       └── ternary/             # optional adapter for ternary-ai agents/prompts
│           ├── adapters.py      # ToolContextProvider, LangChainMemoryAdapter
│           └── prompt_library.py # PromptLibrary (loads ternary-ai/prompts CSVs)
├── examples/
│   ├── openai_agents_sdk.py
│   ├── claude_agents_sdk.py
│   ├── qwen_agents_sdk.py
│   └── google_a2a.py
└── tests/
```

## Optional integrations

`deepcq.integrations.ternary` is an optional adapter layer for using deepcq
as a memory/context backend in agent frameworks. It has no hard dependency
on any framework — both adapters duck-type the interface they mimic:

- **`ToolContextProvider`** — formats a `QueryResult` as a plain string to
  inject into a system prompt or tool call.
- **`LangChainMemoryAdapter`** — duck-types LangChain's `BaseChatMemory`
  (`save_context` / `load_memory_variables` / `clear`), so it can be passed
  directly as a chain's `memory=` argument without installing LangChain.
- **`PromptLibrary`** — loads the CSV prompt templates from the
  [ternary-ai/prompts](https://github.com/ternary-ai/prompts) repo (locally
  or via `urllib` from raw GitHub URLs) into a deepcq schema, so prompt
  templates themselves can be retrieved by intent.

```python
from deepcq import DeepCQ, MemorySchema
from deepcq.integrations.ternary import ToolContextProvider, PromptLibrary

cq = DeepCQ(schema=MemorySchema(sections=["research", "risk-management"]))
provider = ToolContextProvider(cq)
context = provider.get_context("what is my portfolio concentration risk?")
```

See `examples/openai_agents_sdk.py`, `examples/claude_agents_sdk.py`,
`examples/qwen_agents_sdk.py`, and `examples/google_a2a.py` for focused
integration examples. They are designed as SDK wiring references, so you
may need to adjust the exact import path to match the SDK version you have
installed.

## Testing

```bash
pip install -e ".[dev]"
pytest
```

## Moving this into its own repository

This directory is self-contained — nothing outside `deepcq/` is required.
To split it out:

```bash
cp -r deepcq/ /path/to/new/deepcq-repo
cd /path/to/new/deepcq-repo
git init
pip install -e ".[dev]"
pytest
```

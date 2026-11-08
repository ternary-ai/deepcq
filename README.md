# deepcq

**Deep Context Query** — intent-aware memory querying for LLM agents.

*Remember when it matters.*

deepcq decides which parts of an agent's working memory go into the context.
It sits between an agent's memory and its prompt: instead of stuffing an
entire memory blob into every LLM call, it scores which sections of memory
are relevant to the current question and returns only those — within a
context budget, with every inclusion explained and every gap reported.

No embeddings, no vector store, no model call, no external dependencies.
Relevance is computed from the query's own words and figures against text
already in memory, using plain regex and a lightweight stemmer, so the same
memory and the same question always give the same result.

deepcq is a read layer. It works underneath model-driven memory systems
(which decide what to write and keep it correct) and makes the per-question
selection fast, deterministic and auditable.

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
If your chunks already exist (tool results, chat turns), pass a list of
strings instead: indices are positions in that list and no delimiter can
misalign them.
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

`MemorySchema(pinned=[...])` always includes those sections when present,
even below threshold (`sections_pinned`). Pinned sections go first within
the context budget but still obey it: one too large to fit is cut and
reported in `sections_over_budget`.

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

In the sectioned tiers (state, notes, reference) a header must start a
line: a mid-line mention like `"see [status] for uptime"` doesn't start the
`status` section. A section ends at the next *header line* — a declared
section's header, or a header-format line whose name starts with a letter
and has no commas, quotes or brackets, followed by a space or the line end —
so data lines such as
`[AAPL, MSFT]`, `["a", "b"]` or a Markdown link don't cut it short. For
formats with nothing after `{name}` (`"## {name}"`) any line starting with
the prefix is a header. The free-text tiers (cache, history) still score a
section from a mention anywhere in a line (`"USER: [deploy] failed"`), via
`MemorySchema.mention_pattern(section)`.

Returned pieces are joined with a single newline, like the original text,
so the filtered content is never longer than the input.

To rebuild filtered text yourself, `split_sections(text, schema)` splits
text at every header in the schema's format into a preamble plus
`Section(name, text, known)` blocks — including headers the schema doesn't
declare, which `query()` reports in `unknown_sections` (and logs once)
because it can never return their content. `cq.get_section(name)` returns
one stored section by name or prefix, the handler for an agent's "load the
section that was omitted" tool.

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
    sections_pinned: list[str]        # included below threshold via MemorySchema.pinned
    sections_absent: list[str]        # in the schema, but in no tier
    unknown_sections: list[str]       # headers in state that the schema doesn't declare
    fallback_used: bool               # see "Fallback safety"
    fallback_reason: str | None
    sections: dict[str, SectionInfo]  # per section: status, score, chars, description
    sections_over_budget: list[str]   # passed the threshold, cut by max_chars
    tiers_over_budget: list[str]      # "cache" / "history" cut by max_chars
    max_chars: int | None             # the budget this result was built under
    duplicate_sections: dict[str, list[str]]  # section -> tiers holding it twice
    gaps: list[MemoryGap]             # what memory doesn't hold

    def section_map(self, only_omitted=False) -> str
    def gap_note(self) -> str
    def omitted_note(self, reload_hint=None) -> str  # section_map(only_omitted=True) + hint
```

`sections_excluded` lists sections that are present but scored below
threshold — exactly what the agent can ask to reload.

### Context budget (0.4.0)

`query()` returns at most `MemorySchema.max_chars` characters of content —
25,000 by default (about 6,000 English tokens), counting the separators.
Sections that pass the threshold are kept highest score first; a section
that doesn't fit is skipped whole, never truncated, and smaller relevant
sections below it still get in. Then cache and history, if they fit.
Content keeps its usual order. Cut sections are listed in
`sections_over_budget` (free-text tiers in `tiers_over_budget`), logged, and
counted as `budget_cut`. Override per call with `query(..., max_chars=n)`;
`None` turns the budget off. `locate()` takes the same `max_chars`.

### Section descriptions and the section map (0.4.0)

```python
schema = MemorySchema(
    sections=["positions", "risk", "news"],
    descriptions={"positions": "current holdings and sector exposure",
                  "risk": "position limits and drawdown history"},
)
result = cq.query("are we overexposed to tech?")
print(result.section_map(only_omitted=True))
# Not loaded (ask for any of these by name):
# - risk (1,240 chars, not loaded): position limits and drawdown history
```

A description says what a section holds, in the words people ask with. Its
words join the section's scoring vocabulary (turn off with
`score_descriptions=False`), which bridges everyday questions to stored
jargon such as tickers. `result.section_map()` lists every present section
with its status, size and description, so the model can see what exists
without loading it, and ask for a section word matching missed.
`result.sections` holds the same data per section (`SectionInfo`).

### Memory gaps (0.4.0)

When the question asks for something memory doesn't hold, `result.gaps`
says so, and `result.gap_note()` renders it for the agent's context:

```
Not in memory: audit (internal audit findings). No stored memory mentions "10-K".
```

Three kinds: `absent_section` (the question matches a declared section
that isn't stored), `unmatched_anchor` (an exact figure or identifier from
the question appears nowhere in memory) and `no_match` (nothing matched).
The note states facts only; what the agent should do next is up to your
prompt. Gaps are logged at `INFO` and counted in `deepcq.gap_counts()`.

### Duplicate sections (0.4.0)

If one tier holds the same section twice (for example under two
`state` keys), only the first copy is used. deepcq reports it in
`result.duplicate_sections`, warns once per section and tier, and counts it
as `duplicate_section`.

### Usage statistics and health report (0.4.0)

Each `DeepCQ` instance counts, per declared section, how often it was
present, included, excluded, cut by the budget, or asked for while absent.
`cq.usage()` returns the counters; `cq.health_report()` turns them into
findings for an upkeep pass: sections never or always included, oversized,
stale (given `recency` and `max_age`), duplicated, never present, or
requested but missing. Checks that couldn't run are listed in `skipped`.
`report.summary()` prints one finding per line.

**Privacy.** Usage tracking is on by default, so it is safe by design: the
counters hold only counts keyed by declared section names and fixed kind
names — never query text, figures, memory content or anything identifying
a caller — with no per-query records, and they never leave the instance.
For an instance shared by several users, use `health_report(min_count=5)`
to withhold small counts, or `DeepCQ(..., track_usage=False)`. deepcq's log
records likewise carry no query text or figures by default;
`DeepCQ(..., log_query_text=True)` adds them for debugging, along with the
names of undeclared headers found in memory (which are always listed in
`result.unknown_sections`).

### Measuring what word matching misses

`calibrate()` also reports `lexical_misses`: expected sections that shared
no word, phrase or figure with the question. No threshold can reach those,
so the fix is a description or keywords, not tuning.

### Fallback safety

If no section clears the confidence threshold and
`schema.fallback_include_all` is `True` (the default), `query()` returns the
full unfiltered text instead of nothing — this prevents a vague query from
silently discarding memory the agent might actually need. The context
budget still applies: if all of memory is larger than `max_chars`, the
fallback returns the highest-scoring sections that fit and says so in
`fallback_reason`.

The fallback is never silent: the result has `fallback_used=True` and a
`fallback_reason`, a warning is logged on the `"deepcq"` logger, and
`deepcq.fallback_counts()` returns per-kind totals (`no_section_scored`,
`no_section_present`, plus `absent_only`, `budget_cut` and
`duplicate_section` — see above) since process start or the last
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
│   ├── usage.py                 # usage counters + health report
│   ├── vocabulary.py            # SynonymMap, load_pack()
│   ├── vocab_packs/             # built-in synonym packs (finance.json)
│   ├── phrases.py               # extract_phrases() for phrase-aware matching
│   ├── calibration.py           # calibrate() threshold sweep
│   ├── locate.py                # locate() chunk retrieval
│   ├── sections.py              # split_sections(), Section, SectionSplit
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

# Agents: name the omitted sections in the context and give the model a tool
# to load one, instead of letting it assume the data doesn't exist.
context = provider.get_context_with_note("what is my portfolio concentration risk?")
tool = provider.section_tool_spec()       # {"name", "description", "parameters"}
handler = provider.get_section            # call with the tool's `name` argument
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

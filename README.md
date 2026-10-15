# deepcq

**Deep Context Query** — intent-aware memory querying for LLM agents.

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
(default `0.3`) are "included".

### 4. Extracting content (`schema.py`, `regex_gen.py`)

Sections are delimited in stored text by a header, `"[{name}]"` by default
(configurable via `section_header_format`, e.g. `"## {name}"`). Each section
compiles to a regex that captures from its header up to the next header or
end of string — this is how `MemorySchema.extraction_pattern(section)` pulls
a section's block out of raw tier text.

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
```

### Fallback safety

If no section clears the confidence threshold and
`schema.fallback_include_all` is `True` (the default), `query()` returns the
full unfiltered text instead of nothing — this prevents a vague query from
silently discarding memory the agent might actually need.

## Package layout

```
deepcq/
├── pyproject.toml
├── README.md
├── deepcq/                      # the installable package
│   ├── __init__.py              # public API: DeepCQ, MemorySchema, QueryResult, ...
│   ├── core.py                  # DeepCQ orchestrator (query())
│   ├── schema.py                # MemorySchema + extraction regex compilation
│   ├── intent.py                # score_sections()
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

# deepcq 0.3.2 — agent context primitives

Found while integrating deepcq as the context filter of an LLM research
agent: the pieces every agent integration had to rebuild around `query()`.
Builds on the 0.3.1 bug fixes (exact-figure anchors, header boundaries,
absent sections, loud fallback).

## Additions

| Feature | API |
|---|---|
| Pinned sections | `MemorySchema(pinned=[...])`, `QueryResult.sections_pinned` (pins don't suppress the fallback) |
| Schema drift | `QueryResult.unknown_sections` — headers in the state tier the schema doesn't declare, warned once per schema+name |
| Section parser | `split_sections(text, schema) -> SectionSplit(preamble, sections, .unknown, .find())` |
| Section lookup | `DeepCQ.get_section(name, prefix=True)` across state / notes / reference |
| Omitted note | `QueryResult.omitted_note(reload_hint=None)` |
| Reload tool | `ToolContextProvider.get_context_with_note()`, `.section_tool_spec()`, `.get_section()` |
| Pre-chunked locate | `locate(["chunk", ...], query)` — indices are list positions, texts returned exactly |

## Not included (candidates for later)

- A guard against trimming that grows the context (`overhead_chars`).
- Finance anchor pack (filing forms, fiscal periods) alongside the vocab packs.
- `calibrate()` support for `locate()` thresholds.
- A chat-history compaction helper over the History tier.
- Investigate: on an integrator's labeled set the `finance` vocab pack cost
  precision at equal recall, and `phrase_sizes=(2,)`/`(2, 3)` changed no
  decision at all.

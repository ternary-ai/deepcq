# Changelog

## 0.4.1 — 2026-11-13

### Changed

- **No default context budget.** `MemorySchema.max_chars` and `locate()`'s
  `max_chars` now default to `None`, restoring the 0.3.1 behaviour: nothing
  is cut unless you set a budget. The 25,000-character default in 0.4.0 had
  no relation to the caller's model, changed upgraders' output without
  being asked for, and could drop the highest-scoring section whenever it
  alone was larger than the budget. Set `max_chars=n` to turn the budget
  on; cuts are still listed, logged and counted as `budget_cut`.
- `health_report()` skips the oversized check (and lists it in `skipped`)
  unless `max_chars` or `max_section_chars` is set.

## 0.4.0 — 2026-11-13

deepcq decides which parts of an agent's working memory go into the
context. This release makes that selection bounded, explained and honest
about what is missing, derived from Anthropic's
[The components of agent memory](https://www.anthropic.com/research/components-of-agent-memory).
Everything is on by default and deterministic: no model call, and the same
memory and question always give the same result.

### Added

- **Context budget.** `MemorySchema.max_chars` (default 25,000 characters)
  caps what `query()` returns, separators included. Sections that pass the
  threshold are kept highest score first; one that doesn't fit is skipped
  whole, never truncated. Cut sections are listed in
  `QueryResult.sections_over_budget` (cache/history in `tiers_over_budget`),
  logged and counted as `budget_cut`. `query(..., max_chars=n)` overrides
  it; `None` turns it off. `locate()` takes the same `max_chars`.
- **Section descriptions.** `MemorySchema(descriptions={...})`, one line per
  section. Shown in the section map and gap note, and added to the
  section's scoring vocabulary (`score_descriptions=False` turns that off).
- **Section map.** `QueryResult.sections` (`SectionInfo`: status, score,
  size, description) and `QueryResult.section_map(only_omitted=False)`, a
  compact index of present sections so a model can ask for one it needs.
- **Memory gap signal.** `QueryResult.gaps` (`MemoryGap`) and `gap_note()`:
  sections the question asked for that memory doesn't hold, figures or
  identifiers found nowhere in memory, and no match at all. Logged at
  `INFO`, counted in `deepcq.gap_counts()` / `reset_gap_counts()`.
- **Duplicate-section warning.** A section stored twice in one tier is
  reported in `QueryResult.duplicate_sections`, warned once per section and
  tier, and counted as `duplicate_section`. The first copy is still used.
- **Usage statistics and health report.** Per-instance counters
  (`DeepCQ.usage()`, `reset_usage()`) and `DeepCQ.health_report()`:
  sections never or always included, oversized, stale, duplicated, never
  present, or requested but missing, with skipped checks listed. Counters
  hold only counts keyed by declared names — never query text or memory
  content — and never leave the instance. `health_report(min_count=k)`
  withholds small counts for shared instances.
- **Pinned sections.** `MemorySchema(pinned=[...])` keeps sections whenever
  present, even below threshold (`QueryResult.sections_pinned`). They go
  first within the context budget but still obey it, and don't suppress
  the fallback. Pinning an undeclared section raises `ValueError`.
- **Unknown-section report.** `QueryResult.unknown_sections` lists headers
  in the state tier that the schema doesn't declare, so a renamed header
  can't silently drop data. Warned once per schema and name; the warning
  gives a count, and names only with `log_query_text=True`.
- **Section parser and lookup.** `split_sections(text, schema)` returns a
  preamble plus `Section(name, text, known)` blocks; `DeepCQ.get_section()`
  fetches one by name, then prefix, case-insensitively.
- **Omitted-section note and reload tool.** `QueryResult.omitted_note()`
  (the section map's omitted list plus an optional hint), and
  `ToolContextProvider.get_context_with_note()`, `section_tool_spec()` and
  `get_section()` in the ternary integration.
- **Pre-chunked `locate()`.** Pass a list of strings: indices are list
  positions, texts come back exactly as given, and blank entries are
  skipped without shifting indices.
- **Lexical misses.** `calibrate()` reports expected sections that shared
  no word, phrase or figure with the question (`lexical_misses`), which no
  threshold can reach.
- **Digit-led identifiers as anchors.** "10-K", "8-K", "10-Q" are now
  extracted whole from questions and matched exactly.

### Changed

- `query()` returns at most 25,000 characters by default; the
  include-everything fallback is capped by the same budget. Set
  `max_chars=None` for 0.3.1 behaviour.
- `locate()` returns candidates totalling at most 25,000 characters by
  default.
- Log records no longer contain query text: the fallback and `absent_only`
  warnings drop their 80-character excerpt. `DeepCQ(..., log_query_text=True)`
  restores it.
- Each `DeepCQ` instance tracks usage counts (`track_usage=False` to turn
  off).
- Scores change only where the new features add vocabulary: for schemas
  that declare descriptions, and for questions naming a digit-led
  identifier such as "10-K" (previously only "10" was extracted, which
  never matched). Other thresholds need no recalibration.

### Fixed

- `from deepcq import locate` returned the `deepcq.locate` module instead
  of the function once anything had imported the submodule.

## 0.3.1 — 2026-11-01

Bug fixes. Several change scores or returned content by default, so
thresholds tuned against 0.3.0 may need a recalibration pass (see
`calibrate()`).

### Fixed

- **Literal anchors match the exact figure or whole word only.** In 0.3.0 an
  anchor was a plain substring test, and an anchor hit forces a score to
  1.0, so "is 10 a good entry?" force-included every section containing
  "100", "2010" or "2.10".
  - "10" no longer matches "100", "2010", "2.10", "10.5", "10th", "10-K" or
    "10-year". It still matches "top 10", "$10" and the range "10-15%".
  - "15%" no longer matches "115%", "15.5%", "-15%" or "−15%".
  - A leading minus sign is kept on extracted numbers, so "-15%" and "15%"
    are different anchors, and "-15%" doesn't match the range "10-15%".
  - Matching is textual: "1000" doesn't match "1,000", and "15" doesn't
    match "15.0".
- **Section headers end at a word boundary.** `"## SEC"` no longer matches a
  `"## Sector"` header.
- **Sections that no tier contains score 0.0.** Matches on a section's
  name or keywords alone gave it a passing score, and it was listed in
  `sections_included` with no content. Such sections now have
  `ScoreExplanation.present = False` and are listed in the new
  `QueryResult.sections_absent`, never in `sections_included` or
  `sections_excluded`.
- **A query that matches only absent sections returns no section content.**
  It asks about data memory doesn't hold, so it no longer triggers the
  include-all fallback (which would return all of memory).
  `ScoreExplanation.absent_score` keeps the score the section's
  name/keywords gave the query.
- **Headers must start a line in the sectioned tiers** (state, notes,
  reference). A mid-line mention such as "see [status] for uptime" started
  the `status` section there, so the wrong text was scored and returned
  and the real section was lost. The free-text tiers (cache, history)
  still score a section from a mention anywhere in a line
  ("USER: [deploy] failed"), via the new `MemorySchema.mention_pattern()`.
- **Data lines no longer end a section.** With the default `"[{name}]"`
  format any line starting with `[` ended the section, cutting off content
  after lines such as `[AAPL, MSFT]`, `["a", "b"]` or a Markdown link. A
  section now ends at the next header line: a declared section's header,
  or a header-format name that starts with a letter and has no commas,
  quotes or brackets, followed by a space or the end of the line. Formats
  with nothing after `{name}` (`"## {name}"`) are unchanged.
- **`reduction_pct` is never negative.** Returned pieces are joined with a
  single newline, like the original text (previously a blank line), so the
  filtered content can't be longer than the input.

### Added

- **Loud fallback.** When `fallback_include_all` returns everything
  unfiltered, the result has `fallback_used = True` and a
  `fallback_reason`, and a warning is logged on the `"deepcq"` logger (the
  query is cut to 80 characters). `deepcq.fallback_counts()` returns
  process-wide totals by kind (`no_section_scored`, `no_section_present`,
  `absent_only`); `deepcq.reset_fallback_counts()` zeroes them.

## 0.3.0 — 2026-10-15

Synonym vocabularies, phrase matching, per-tier weighting, threshold
calibration (`calibrate()`) and per-tier vocabulary caching.

## 0.2.1 — 2026-10-15

Recency decay and per-section score explanations.

## 0.2.0 — 2026-10-15

Literal-anchor matching and the `locate()` chunk-retrieval API.

## 0.1.0 — 2026-10-15

Initial release.

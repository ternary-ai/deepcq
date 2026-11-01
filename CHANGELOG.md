# Changelog

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

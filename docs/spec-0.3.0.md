# deepcq 0.3.0 feature spec

Five additions, grouped because they share a constraint: every one of them
must be **opt-in and additive**. deepcq has real integrators with thresholds already
tuned against production fixtures. None of these features may change a
`QueryResult.intent` score for a schema that doesn't opt into the new field,
and each section below states explicitly why its default reduces to
today's behavior.

Target release: 0.3.0 (minor — additive fields only, no signature removals).

---

## 1. Synonym / domain-vocabulary plugins

### Problem

`intent.py` scores a section by token overlap between the query and that
section's own vocabulary (name + `MemorySchema.keywords` + stored text).
When the query uses a different word than the stored text does — user asks
about "eps", the section stores "earnings per share" — there's no overlap
unless the integrator manually lists the synonym in `keywords` — a
hand-maintained keyword table per section, per integrator, forever.

### Design

New module `deepcq/vocabulary.py`:

```python
@dataclass(frozen=True)
class SynonymMap:
    """A set of terms that should be treated as mutually interchangeable
    for scoring purposes. Expansion is symmetric: if "eps" and "earnings
    per share" are in the same cluster, each expands to include the other."""

    _clusters: tuple[frozenset[str], ...]

    @classmethod
    def from_dict(cls, mapping: dict[str, Sequence[str]]) -> "SynonymMap":
        """mapping keys ∪ their listed synonyms form one cluster each.
        Overlapping clusters (a term appears under two keys) are merged."""

    def expand(self, token: str) -> frozenset[str]:
        """All terms in token's cluster, tokenized+stemmed, including
        token itself. Empty-frozenset-of-just-token if token is in no
        cluster (i.e. a no-op for unknown terms)."""

    def __or__(self, other: "SynonymMap") -> "SynonymMap":
        """Compose two packs (e.g. a built-in pack + a caller's own terms)."""


def load_pack(name: str) -> SynonymMap:
    """Built-in domain packs shipped as deepcq/vocab_packs/{name}.json —
    flat {canonical: [synonym, ...]} files. Ships with "finance" first
    as the reference pack;
    "devops" and "support" are candidates for a follow-up once there's a
    second real integrator to validate the term list against."""
```

Multi-word synonyms (`"capex" -> "capital expenditure"`) tokenize into
their own unigram set on expansion — a synonym pack gets you token-level
recall, not phrase-level precision. That's what §2 is for; the two
features compose (a pack can list a multi-word canonical form, and once
phrase-aware matching is on, an exact phrase match on the expanded form
scores higher than a scattered unigram overlap would).

### Wiring

`MemorySchema` gains one field:

```python
vocabulary: SynonymMap | None = None   # default: no expansion, current behavior
```

In `score_sections_detailed()` (`intent.py`), both `query_tokens` and each
section's `vocab` are expanded through `schema.vocabulary.expand()` before
intersecting — symmetric, so it doesn't matter whether the query or the
stored text uses the jargon form:

```python
query_tokens = set(tokenize(query, include_stems=True))
if schema.vocabulary is not None:
    for t in list(query_tokens):
        query_tokens |= schema.vocabulary.expand(t)
```
(same expansion applied to `vocab` after it's built from name/keywords/text).

### Backward compatibility

`vocabulary=None` is the dataclass default; the expansion loop is
skipped entirely, so `query_tokens`/`vocab` are byte-identical to today.
Zero behavior change for every existing caller until it opts in.

### Migration note for integrators

Once shipped, hand-maintained per-section synonym lists in `keywords` can
likely shrink or disappear in favor of `load_pack("finance") | SynonymMap.from_dict({...app-specific extras...})`.

### Tests

`tests/test_vocabulary.py`: pack loading, symmetric expansion, cluster
merging on overlapping keys, `__or__` composition. Extend
`tests/test_intent.py` with a schema that only resolves via a synonym
(replicates the eps/capex case without a `keywords` dict) plus a
`vocabulary=None` parity test against an existing fixture.

---

## 2. Phrase-aware matching

### Problem

Scoring is pure bag-of-words. "moving average" and "average moving" score
identically; a query for "moving average" gets partial credit from any
section that happens to mention "moving" or "average" separately, which is
exactly the kind of incidental overlap integrators guard against today by
raising thresholds on free-form prose sections.

### Design

New module `deepcq/phrases.py`. First, `tokens.py` needs an
order-preserving variant — the existing `tokenize()` interleaves each
token's stem right after it in the output list specifically so callers get
both forms, which destroys positional adjacency. Add:

```python
# tokens.py
def content_words(text: str) -> list[str]:
    """Lowercase, stopword-filtered words in original order — no stemming,
    no dedup. The adjacency-preserving counterpart to tokenize(); phrases.py
    is the only current caller."""
```

```python
# phrases.py
def extract_phrases(text: str, sizes: tuple[int, ...] = (2,)) -> set[str]:
    """Space-joined n-grams of *stemmed* consecutive content words, for
    the given n-gram sizes. Stemmed (not surface-form) so "running average"
    and "run averages" land on the same phrase token — trades a small
    amount of phrase precision for consistency with how unigram matching
    already works. Stopwords are removed before n-gramming (so "cost of
    goods sold" yields "cost goods", "goods sold" — same
    stopword-dropping the unigram path already does)."""
```

### Scoring

`score_sections_detailed()` computes `matched_phrases = query_phrases &
section_phrases` and applies it as a **bonus, never a penalty** — same
non-regression rule the existing literal-anchor mechanism already follows
(anchors.py's docstring: "this only ever raises a score"):

```python
phrase_bonus = min(1.0, schema.phrase_boost * len(matched_phrases))
score = min(1.0, token_score + phrase_bonus)
```

`ScoreExplanation` gains `matched_phrases: list[str] = field(default_factory=list)`.

### Schema fields

```python
phrase_sizes: tuple[int, ...] = ()     # () = feature off (default)
phrase_boost: float = 0.25             # per matched phrase, capped at 1.0 total
```

### Backward compatibility

`phrase_sizes=()` is the default. `extract_phrases()` on an empty sizes
tuple returns `set()`, `matched_phrases` is always empty,
`phrase_bonus` is always `0.0`, so `score == token_score` exactly —
identical to pre-0.3.0 scoring. This is the one feature in this batch that
*does* change numeric scores when enabled, so it's opt-in per schema
rather than symmetric-and-silent like §1; an integrator turning it on
should expect to re-run calibration (§4) before shipping the new
threshold.

### Open question for implementation review

Generating phrases from stemmed-only content words (vs. also generating a
surface-form phrase set, doubling storage) is a simplicity/recall
tradeoff — flagging rather than deciding, since it's cheap to change
before the first release and expensive to change after (numeric score
shift for anyone who's turned it on).

### Tests

`tests/test_phrases.py`: n-gram extraction at various sizes, stopword
interaction, stemming interaction. Extend `tests/test_intent.py`: a case
where unigram overlap alone stays under threshold but the phrase bonus
pushes it over; a `phrase_sizes=()` parity test.

---

## 3. Per-tier weighting

### Problem

`score_sections_detailed()` currently concatenates a section's matched
text across **all** tiers into one blob before tokenizing
(`intent.py:54-62`) — tier identity is discarded before scoring even
starts. There's no way to say "a match in `state` should count for more
than the same match surfacing in stale `history`."

### Design

This needs tier identity preserved through scoring, which the current
loop structure doesn't have — the biggest structural change of the five.
`score_sections_detailed()`'s inner loop changes from
`for text in tier_texts.values()` to `for tier_name, text in
tier_texts.items()`, building a per-tier vocab set instead of one merged
blob.

**Scoring formula — weighted union, chosen to reduce to the current
formula exactly at default weights:**

For each query token, its contribution is the highest tier-weight among
tiers where it was found (not a sum across tiers — a token shouldn't count
twice for appearing in two tiers, it should count at its best-weighted
occurrence):

```python
def contribution(token: str) -> float:
    found_in = [w for tier, w in tier_weights.items() if token in tier_vocab[tier]]
    if token in name_and_keyword_vocab:
        found_in.append(schema_weight)   # default 1.0, tier-independent
    return min(1.0, max(found_in, default=0.0))

score = sum(contribution(t) for t in query_tokens) / len(query_tokens)
```

At `tier_weights = {tier: 1.0 for tier in ALL_TIERS}` (or `None`, see
below), `contribution(t)` is `1.0` if found anywhere, `0.0` otherwise —
`sum(...) / len(query_tokens)` is then arithmetically identical to
today's `len(matched_tokens) / len(query_tokens)`. This is the
correctness argument for backward compatibility, not just "defaults
happen to match."

Literal-anchor matching gets the same treatment for consistency: an
anchor's `literal_hit` is weighted by the max tier-weight of the tier it
was found in, rather than unconditionally forcing `score = 1.0` regardless
of which tier it came from.

### Schema fields

```python
# schema.py — new constant, exported from __init__.py
TIER_NAMES = ("state", "cache", "history", "notes", "reference")

# MemorySchema
tier_weights: dict[str, float] | None = None   # None = all tiers weight 1.0 (current behavior)
schema_weight: float = 1.0                     # weight for section name/keywords vocab (tier-independent)
```

### `ScoreExplanation` addition

```python
tier_contributions: dict[str, float] = field(default_factory=dict)  # tier_name -> contribution to score, for debug UIs
```

### Scope note

`locate()` builds a one-shot `DeepCQ` and stores every chunk in the
`state` tier only (`locate.py:53-54`) — tier weighting has nothing to act
on there. Out of scope for this feature; `locate()`'s `recency` parameter
is the existing analogous per-chunk weighting mechanism for that code path.

### Backward compatibility

`tier_weights=None` short-circuits to the all-1.0 case above — proven
identical to current scoring, not just defaulted to look similar.

### Tests

`tests/test_intent.py`: a fixture where the same token appears in two
tiers with different weights, asserting the max-not-sum rule; a
`tier_weights=None` parity test against every existing fixture in that
file (regression net for the loop-structure rewrite, since that's the
riskiest part of this change).

---

## 4. Threshold calibration helper

### Problem

Integrators end up with several per-schema confidence thresholds (e.g.
0.12, 0.15, 0.2, 0.25), each chosen empirically against their own test
fixtures. Every integrator either does this manual sweep-and-eyeball process or skips it
and picks a threshold that "feels right."

### Deviation from the originally-sketched signature

The requested shape was `calibrate(schema, query_log, expected_sections)`.
In practice, relevance is state-dependent — "should section X be
included" only means something relative to what's actually stored in the
tiers *at query time*, and different queries in a calibration set
typically need different memory state (that's exactly what integrator
test fixtures already do: each test case sets up its own `DeepCQ`
instance). A flat `schema` + `query_log` doesn't
carry that. The spec below threads memory-state through a factory per
example instead — closer to how integrators already write test fixtures,
and it lets `calibrate()` reuse `DeepCQ.query()` verbatim rather than
re-deriving scoring logic.

### Design

New module `deepcq/calibration.py`:

```python
@dataclass
class LabeledExample:
    """One (memory-state, query, expected-sections) calibration case."""
    build: Callable[[], DeepCQ]   # returns a fully populated DeepCQ for this case
    query: str
    expected_sections: set[str]   # section names a correct filter should include


@dataclass
class ThresholdStats:
    threshold: float
    precision: float
    recall: float
    f1: float
    true_positives: int
    false_positives: int
    false_negatives: int


@dataclass
class CalibrationReport:
    stats: list[ThresholdStats]          # one per swept threshold, ascending
    best: ThresholdStats                 # see selection rule below

    def summary(self) -> str:
        """Pretty-printed table: threshold | precision | recall | f1."""


def calibrate(
    examples: Sequence[LabeledExample],
    thresholds: Sequence[float] = tuple(round(0.02 * i, 2) for i in range(1, 26)),  # 0.02..0.50
    min_precision: float | None = None,
) -> CalibrationReport:
    """
    For each threshold t and each example:
      cq = example.build()
      result = cq.query(example.query)          # schema's own threshold is irrelevant here —
      included = {s for s, sc in result.intent.items() if sc >= t}   # we re-threshold result.intent directly
      TP = |included & expected|, FP = |included - expected|, FN = |expected - included|

    Aggregates TP/FP/FN across all examples at each t (micro-averaged —
    a query log with skewed section frequency shouldn't let one
    over-represented section dominate a macro average).

    `best` is the threshold with the highest F1; ties broken toward the
    *higher* threshold (prefer the more conservative, smaller-context
    option among equally-accurate choices — matches the reduction-focused
    goal of the library).

    If `min_precision` is set, `best` is instead the highest-recall
    threshold with precision >= min_precision, falling back to the
    unconstrained F1-best (with `CalibrationReport.constraint_met = False`)
    if no threshold clears the floor.
    """
```

Note `result.intent` already carries every section's raw score regardless
of `schema.confidence_threshold` — `calibrate()` calls
`DeepCQ.query()` exactly once per example per... no, once per example
*total*: the threshold sweep re-thresholds the same `result.intent` dict
25 times without re-running scoring, so a calibration run over a
log of N examples costs N query() calls, not N × len(thresholds).

`apply_fallback` is deliberately **not** simulated during the sweep (raw
`>= t` thresholding only) — `fallback_include_all` exists to protect
against zero-signal queries in production, and folding it into precision/
recall would let a bad threshold hide behind the fallback safety net
instead of being visible in the numbers.

### Backward compatibility

Purely additive — a new module, no changes to `DeepCQ`, `MemorySchema`,
or scoring. Nothing to break.

### Migration note for integrators

Existing per-case test fixtures are most of the way to `LabeledExample`s
already; the remaining work is adding `expected_sections` labels per case
and running `calibrate()` in place of hand-picked thresholds — e.g. a
`scripts/calibrate_thresholds.py` re-run whenever the schema's sections
change.

### Tests

`tests/test_calibration.py`: a synthetic query log with a known-correct
threshold, asserting `calibrate()` finds it; precision/recall arithmetic
on a hand-computed small example; `min_precision` constraint satisfied
and unsatisfiable cases; single-call-to-query()-per-example asserted via
a call-counting `DeepCQ` subclass or monkeypatch (regression guard against
the sweep accidentally becoming O(examples × thresholds) in scoring
work later).

---

## 5. Cached per-tier vocabulary

### Problem

`score_sections_detailed()` re-tokenizes every section's matched text on
every `query()` call, unconditionally (`intent.py:54-62`) — for a
long-running agent with a large, mostly-static `LongTermStore` or
`ReferenceData` tier (the two tiers explicitly designed to be
"persistent"), this repeats identical regex-search-and-tokenize work on
every single query even when nothing in that tier has changed since the
last one.

### Design

**Where the cache lives:** on the tier objects themselves, not on
`DeepCQ` — a tier's content and mutation points are already fully known
locally to that class, and this keeps the cache correctly invalidated
even if a tier instance is ever shared across `DeepCQ` instances.

**Invalidation key:** a monotonic `version` counter per tier, bumped by
every mutating method. This is a small, mechanical addition to each tier
class:

| File | Class | Mutators to instrument |
|---|---|---|
| `memory/state.py` | `AppState` | `set` |
| `memory/session.py` | `SessionCache` | `put` (only on actual change — it already no-ops on identical value, which conveniently means the version bump belongs *inside* the existing dedup check), `clear` |
| `memory/session.py` | `History` | `add`, `clear` |
| `memory/store.py` | `LongTermStore` | `append` |
| `memory/store.py` | `ReferenceData` | `set` |

**What's cached:** not raw text (cheap) but the result of `pat.search()`
+ `tokenize()` for a given `(schema, section)` — the actual expensive
step. Cache lives on the tier, keyed by `(schema.schema_id(), section)`
(schema already exposes `schema_id()` for exactly this kind of
cross-schema cache-key use, per `schema.py:67-68`), storing
`(matched_text, vocab_tokens)`:

```python
# added to each tier class
def _vocab_entry(
    self, schema_id: int, section: str, compute: Callable[[], tuple[str, list[str]]]
) -> tuple[str, list[str]]:
    key = (schema_id, section)
    if self._vocab_version_seen.get(key) != self._version:
        self._vocab_cache[key] = compute()
        self._vocab_version_seen[key] = self._version
    return self._vocab_cache[key]
```

`score_sections_detailed()` calls this instead of doing the
`pat.search`/`tokenize` inline, passing a `compute` closure that does
exactly what the current inline code does. On a cache hit, both the regex
search *and* the tokenize call are skipped — not just the tokenize step —
since the closure that would do both is never invoked.

No LRU/eviction needed: the key space is `(schema × section)`, which is
small and stable for a given integration (bounded by however many
distinct `MemorySchema`s and section names an app actually uses, not by
query volume) — unlike a per-query cache, this doesn't grow with traffic.

### Backward compatibility

Purely an internal performance change — `QueryResult` output is
unaffected. The correctness risk is entirely in invalidation being wrong
(stale cache), not in behavior changing for anyone.

### Tests

`tests/test_caching.py`:
- Parity: for every existing fixture across `test_core.py`/
  `test_intent.py`, `QueryResult` is identical whether or not this change
  is present (run the full existing suite unmodified — it should stay
  green with zero test edits, which is itself the strongest parity check).
- Invalidation: score changes correctly after `cq.state.set(...)`
  mutates a section already cached from a prior `query()` call.
- Cache-hit verification: wrap `tokenize()` with a call counter across
  repeated `query()` calls against unchanged memory, asserting the count
  does *not* grow linearly with the number of `query()` calls (the actual
  perf claim, not just "still correct").
- `SessionCache.put()`'s existing no-op-on-identical-value path is
  covered by an explicit test that the version does *not* bump (and the
  cache is *not* invalidated) when a caller re-puts the same value.

---

## Cross-cutting notes

- **Ordering:** §5 (caching) has zero interaction risk with §1–4 and is
  the safest to land first — pure performance, no schema/API surface. §3
  (tier weighting) is the riskiest given the `intent.py` loop restructure
  and should land with the widest parity-test net before §1/§2 build on
  top of the same function.
- **Versioning:** all five are additive fields/modules → 0.3.0 (minor),
  not 1.0.0. None of `DeepCQ.__init__`, `MemorySchema`'s existing fields,
  or `QueryResult`'s existing fields change shape — only grow new
  optional ones.
- **`__init__.py` exports** to add: `SynonymMap`, `load_pack`,
  `LabeledExample`, `ThresholdStats`, `CalibrationReport`, `calibrate`,
  `TIER_NAMES`.

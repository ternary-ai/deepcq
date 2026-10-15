"""Synonym / domain-vocabulary maps for scoring.

A SynonymMap groups terms into clusters of interchangeable terms. Scoring
expands each section's vocabulary with every cluster whose term occurs in
that section's text (or name/keywords), so a query using a different word
than the stored text does — "eps" vs "earnings per share" — still overlaps:

    from deepcq import MemorySchema, SynonymMap, load_pack

    vocab = load_pack("finance") | SynonymMap.from_dict({"arr": ["annual recurring revenue"]})
    schema = MemorySchema(sections=["fundamentals"], vocabulary=vocab)
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field
from importlib import resources
from typing import Iterable, Sequence
from .tokens import content_words, stem, tokenize

_TermKey = tuple[str, ...]


def _term_key(term: str) -> _TermKey:
    """Stemmed content words of a term — the form terms are matched on."""
    key = tuple(stem(w) for w in content_words(term))
    if not key:
        raise ValueError(f"synonym term {term!r} has no content words (only stopwords/punctuation)")
    return key


def _merge_clusters(groups: Iterable[Iterable[str]]) -> tuple[frozenset[str], ...]:
    """Union groups that share a term (compared by stemmed key)."""
    clusters: list[set[str]] = []
    keys: list[set[_TermKey]] = []
    for group in groups:
        terms = {t.strip().lower() for t in group if t.strip()}
        group_keys = {_term_key(t) for t in terms}
        overlapping = [i for i, ks in enumerate(keys) if ks & group_keys]
        for i in reversed(overlapping):
            terms |= clusters.pop(i)
            group_keys |= keys.pop(i)
        clusters.append(terms)
        keys.append(group_keys)
    return tuple(frozenset(c) for c in clusters if c)


@dataclass(frozen=True)
class SynonymMap:
    """A set of term clusters treated as mutually interchangeable for
    scoring. Expansion is symmetric: if "eps" and "earnings per share" are
    in the same cluster, each expands to include the other.

    Multi-word terms expand to their unigram tokens — token-level recall,
    not phrase-level precision (that is what phrase matching is for)."""

    _clusters: tuple[frozenset[str], ...] = ()
    _index: dict[_TermKey, int] = field(init=False, repr=False, compare=False)
    _expansions: tuple[frozenset[str], ...] = field(init=False, repr=False, compare=False)
    _key_lengths: tuple[int, ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        clusters = _merge_clusters(self._clusters)
        index: dict[_TermKey, int] = {}
        expansions: list[frozenset[str]] = []
        for i, cluster in enumerate(clusters):
            tokens: set[str] = set()
            for term in cluster:
                index[_term_key(term)] = i
                tokens.update(tokenize(term, include_stems=True))
            expansions.append(frozenset(tokens))
        object.__setattr__(self, "_clusters", clusters)
        object.__setattr__(self, "_index", index)
        object.__setattr__(self, "_expansions", tuple(expansions))
        object.__setattr__(self, "_key_lengths", tuple(sorted({len(k) for k in index})))

    @classmethod
    def from_dict(cls, mapping: dict[str, Sequence[str]]) -> "SynonymMap":
        """Each key plus its listed synonyms forms one cluster. Overlapping
        clusters (a term appears under two keys) are merged."""
        return cls(tuple(frozenset([key, *syns]) for key, syns in mapping.items()))

    @property
    def clusters(self) -> tuple[frozenset[str], ...]:
        return self._clusters

    def expand(self, token: str) -> frozenset[str]:
        """All terms in token's cluster, tokenized+stemmed, including token
        itself. Just {token} if token is in no cluster (a no-op for unknown
        terms). Only single-word terms can match a single token; use
        expand_text() to also catch multi-word terms."""
        for key in ((token,), (stem(token),)):
            i = self._index.get(key)
            if i is not None:
                return self._expansions[i] | {token}
        return frozenset({token})

    def expand_text(self, text: str) -> frozenset[str]:
        """Union of the expansions of every cluster term (single- or
        multi-word) occurring in `text`. Empty if none occur."""
        if not self._index:
            return frozenset()
        stems = [stem(w) for w in content_words(text)]
        found: set[int] = set()
        for n in self._key_lengths:
            for i in range(len(stems) - n + 1):
                cluster = self._index.get(tuple(stems[i:i + n]))
                if cluster is not None:
                    found.add(cluster)
        return frozenset().union(*(self._expansions[i] for i in found))

    def __or__(self, other: "SynonymMap") -> "SynonymMap":
        """Compose two maps (e.g. a built-in pack + a caller's own terms)."""
        if not isinstance(other, SynonymMap):
            return NotImplemented
        return SynonymMap(self._clusters + other._clusters)


def available_packs() -> list[str]:
    """Names of the built-in vocabulary packs."""
    pack_dir = resources.files("deepcq") / "vocab_packs"
    return sorted(p.name[:-5] for p in pack_dir.iterdir() if p.name.endswith(".json"))


def load_pack(name: str) -> SynonymMap:
    """Load a built-in domain pack (deepcq/vocab_packs/{name}.json, a flat
    {canonical: [synonym, ...]} file)."""
    packs = available_packs()
    if name not in packs:
        raise ValueError(f"unknown vocabulary pack {name!r}; available: {packs}")
    raw = (resources.files("deepcq") / "vocab_packs" / f"{name}.json").read_text(encoding="utf-8")
    return SynonymMap.from_dict(json.loads(raw))

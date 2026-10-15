"""Phrase (n-gram) extraction for phrase-aware scoring."""
from __future__ import annotations
from .tokens import content_words, stem


def extract_phrases(text: str, sizes: tuple[int, ...] = (2,)) -> set[str]:
    """
    Space-joined n-grams of *stemmed* consecutive content words, for the
    given n-gram sizes. Stemmed (not surface-form) so "running average" and
    "run averages" land on the same phrase token — consistent with how
    unigram matching already works. Stopwords are removed before
    n-gramming, so "cost of goods sold" yields "cost good", "good sold".

    An empty `sizes` tuple returns an empty set (phrase matching off).
    """
    if not sizes:
        return set()
    stems = [stem(w) for w in content_words(text)]
    phrases: set[str] = set()
    for n in sizes:
        for i in range(len(stems) - n + 1):
            phrases.add(" ".join(stems[i:i + n]))
    return phrases

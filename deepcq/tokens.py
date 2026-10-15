"""Query tokenizer and suffix-strip stemmer. No external dependencies."""
from __future__ import annotations
import re

_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "it", "its",
    "this", "that", "these", "those", "i", "me", "my", "we", "our",
    "you", "your", "he", "she", "him", "her", "his", "hers", "they",
    "them", "their", "what", "which", "who", "whom", "how", "when",
    "where", "why", "all", "any", "both", "each", "few", "more", "most",
    "other", "some", "such", "no", "not", "only", "same", "so", "than",
    "too", "very", "just", "about", "up", "down", "out", "if", "then",
    "as", "also", "into", "through", "during", "before", "after",
    "above", "below", "between", "while", "although", "because", "since",
    "until", "unless", "however", "therefore", "there", "here", "now",
    "show", "tell", "get", "give", "let", "make", "see", "know", "go",
    "come", "think", "say", "want", "look", "use", "find", "take",
    "much", "many", "lot", "like", "new", "good", "right", "old",
    "long", "little", "own", "first", "last", "next", "back", "over",
    "again", "once", "am", "s", "re", "ve", "ll", "d", "please",
})

# Ordered longest-first so greedier matches win.
# "es" is intentionally omitted: "spikes"→"spike" via "s" rule is correct;
# "es" would give "spik" which is worse for content matching.
_SUFFIX_RULES: list[tuple[str, str]] = [
    ("ational", "ate"), ("tional", "tion"), ("izing", "ize"),
    ("ising", "ise"), ("ization", "ize"), ("isation", "ise"),
    ("nesses", "ness"), ("iness", "y"), ("ations", "ate"),
    ("tion", ""), ("ions", ""), ("ing", ""), ("ous", ""),
    ("ive", ""), ("ful", ""), ("ness", ""), ("less", ""),
    ("able", ""), ("ible", ""), ("ment", ""), ("ance", ""),
    ("ence", ""), ("ion", ""), ("ies", "y"), ("ves", "f"),
    ("ed", ""), ("er", ""), ("ly", ""), ("s", ""),
]

_MIN_ROOT = 3
_VOWELS = frozenset("aeiou")


def _dedouble(root: str) -> str:
    """Remove a doubled terminal consonant produced by suffix stripping.

    e.g. "runn" (from running) → "run", "bugg" (from bugged) → "bug".
    """
    if len(root) >= 2 and root[-1] == root[-2] and root[-1] not in _VOWELS:
        return root[:-1]
    return root


def stem(token: str) -> str:
    """Return a suffix-stripped root form with doubled-consonant cleanup."""
    for suffix, replacement in _SUFFIX_RULES:
        if token.endswith(suffix) and len(token) - len(suffix) >= _MIN_ROOT:
            root = token[: len(token) - len(suffix)] + replacement
            if replacement == "":
                root = _dedouble(root)
            return root
    return token


def tokenize(text: str, include_stems: bool = True) -> list[str]:
    """
    Lowercase, strip punctuation, drop stopwords.
    With include_stems=True each unique token is followed by its stemmed
    form (if different), giving the caller both surface and root forms.
    """
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9]*", text.lower())
    seen: set[str] = set()
    result: list[str] = []
    for w in words:
        if w in _STOPWORDS or len(w) < 2:
            continue
        if w not in seen:
            seen.add(w)
            result.append(w)
        if include_stems:
            s = stem(w)
            if s and s != w and s not in seen and len(s) >= _MIN_ROOT:
                seen.add(s)
                result.append(s)
    return result

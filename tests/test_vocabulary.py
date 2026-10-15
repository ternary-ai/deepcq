import pytest

from deepcq.vocabulary import SynonymMap, available_packs, load_pack


def test_load_finance_pack():
    assert "finance" in available_packs()
    pack = load_pack("finance")
    assert "earnings" in pack.expand("eps")
    assert "capital" in pack.expand("capex")


def test_unknown_pack_raises():
    with pytest.raises(ValueError, match="unknown vocabulary pack"):
        load_pack("no-such-pack")


def test_expansion_is_symmetric():
    vocab = SynonymMap.from_dict({"eps": ["earnings per share"]})
    assert {"earnings", "per", "share"} <= vocab.expand("eps")
    assert "eps" in vocab.expand_text("quarterly earnings per share rose")
    # Single-word synonyms expand from either side via expand().
    vocab = SynonymMap.from_dict({"revenue": ["sales", "turnover"]})
    assert "revenue" in vocab.expand("turnover")
    assert "turnover" in vocab.expand("revenue")


def test_expand_unknown_token_is_noop():
    vocab = SynonymMap.from_dict({"eps": ["earnings per share"]})
    assert vocab.expand("latency") == frozenset({"latency"})
    assert vocab.expand_text("latency spiked") == frozenset()


def test_partial_multiword_term_does_not_expand():
    # "share" alone is not the term "earnings per share" — no expansion.
    vocab = SynonymMap.from_dict({"eps": ["earnings per share"]})
    assert vocab.expand("share") == frozenset({"share"})
    assert "eps" not in vocab.expand_text("share price")


def test_expansion_matches_stemmed_forms():
    vocab = SynonymMap.from_dict({"capex": ["capital expenditure"]})
    assert "capex" in vocab.expand_text("capital expenditures increased")


def test_overlapping_keys_merge_into_one_cluster():
    vocab = SynonymMap.from_dict({"revenue": ["sales"], "turnover": ["sales"]})
    assert len(vocab.clusters) == 1
    assert "turnover" in vocab.expand("revenue")


def test_or_composes_and_merges():
    a = SynonymMap.from_dict({"eps": ["earnings per share"]})
    b = SynonymMap.from_dict({"revenue": ["sales"], "eps": ["per share earnings"]})
    combined = a | b
    assert len(combined.clusters) == 2
    assert "eps" in combined.expand_text("per share earnings")
    assert "earnings" in combined.expand("eps")
    assert "sales" in combined.expand("revenue")


def test_stopword_only_term_rejected():
    with pytest.raises(ValueError, match="no content words"):
        SynonymMap.from_dict({"eps": ["of the"]})

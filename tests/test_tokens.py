from deepcq.tokens import stem, tokenize


def test_stem_removes_plural():
    assert stem("errors") == "error"
    assert stem("metrics") == "metric"
    assert stem("services") == "service"


def test_stem_removes_ing():
    assert stem("running") == "run"
    assert stem("crashing") == "crash"
    assert stem("stopping") == "stop"


def test_stem_identity_short_words():
    assert stem("cpu") == "cpu"
    assert stem("db") == "db"


def test_tokenize_drops_stopwords():
    tokens = tokenize("what are the errors in the system")
    assert "what" not in tokens
    assert "the" not in tokens
    assert "errors" in tokens or "error" in tokens


def test_tokenize_includes_stems():
    tokens = tokenize("any memory spikes?", include_stems=True)
    assert "memory" in tokens
    assert "spike" in tokens


def test_tokenize_dedupes():
    tokens = tokenize("error errors errored")
    assert len(tokens) == len(set(tokens))


def test_tokenize_no_stems():
    tokens = tokenize("cpu spikes", include_stems=False)
    assert "cpu" in tokens
    assert "spikes" in tokens
    assert "spike" not in tokens

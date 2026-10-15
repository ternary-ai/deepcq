from deepcq.phrases import extract_phrases


def test_bigrams_default():
    assert extract_phrases("moving average crossover") == {"mov average", "average crossov"}


def test_multiple_sizes():
    phrases = extract_phrases("moving average crossover", sizes=(2, 3))
    assert "mov average crossov" in phrases
    assert "mov average" in phrases


def test_empty_sizes_returns_empty_set():
    assert extract_phrases("moving average crossover", sizes=()) == set()


def test_text_shorter_than_n():
    assert extract_phrases("average", sizes=(2,)) == set()


def test_stopwords_dropped_before_ngramming():
    assert extract_phrases("cost of goods sold") == {"cost good", "good sold"}


def test_stemming_unifies_surface_forms():
    assert extract_phrases("running averages") == extract_phrases("run average")


def test_order_matters():
    assert extract_phrases("moving average") != extract_phrases("average moving")

from mini_bpe.encoder import apply_merges, encode, tokens_to_ids
def test_apply_merges():
    merges = [
        ("l", "o"),
        ("lo", "w"),
    ]
    assert apply_merges("low", merges) == ["low"]
def test_apply_merges_partial_word():
    merges = [
        ("l", "o"),
        ("lo", "w"),
    ]
    assert apply_merges("lower", merges) == ["low", "e", "r"]
def test_tokens_to_ids():
    vocab = {
        "l": 0,
        "o": 1,
        "w": 2,
        "e": 3,
        "r": 4,
        "lo": 5,
        "low": 6,
    }
    assert tokens_to_ids(["low", "e", "r"], vocab) == [6, 3, 4]
def test_encode():
    merges = [
        ("l", "o"),
        ("lo", "w"),
    ]
    vocab = {
        "l": 0,
        "o": 1,
        "w": 2,
        "e": 3,
        "r": 4,
        "lo": 5,
        "low": 6,
    }
    assert encode("lower", merges, vocab) == [6, 3, 4]
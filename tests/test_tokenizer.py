import pytest
from mini_bpe import BPETokenizer
def test_round_trip():
    corpus = "low lower lowest"
    tokenizer = BPETokenizer()
    tokenizer.train(corpus, vocab_size=20)
    ids = tokenizer.encode(corpus)
    assert tokenizer.decode(ids) == corpus
def test_empty_text():
    tokenizer = BPETokenizer()
    tokenizer.train("abc abc", vocab_size=5)
    assert tokenizer.encode("") == []
    assert tokenizer.decode([]) == ""
def test_untrained_tokenizer():
    tokenizer = BPETokenizer()
    with pytest.raises(RuntimeError):
        tokenizer.encode("hello")
def test_empty_corpus():
    tokenizer = BPETokenizer()
    with pytest.raises(ValueError):
        tokenizer.train("", vocab_size=10)
def test_invalid_vocab_size():
    tokenizer = BPETokenizer()
    with pytest.raises(ValueError):
        tokenizer.train("abc", vocab_size=0)
def test_vocab_too_small():
    tokenizer = BPETokenizer()
    with pytest.raises(ValueError):
        tokenizer.train("abc", vocab_size=2)
def test_unknown_character():
    tokenizer = BPETokenizer()
    tokenizer.train("abc abc", vocab_size=5)
    with pytest.raises(ValueError):
        tokenizer.encode("z")
def test_save_load(tmp_path):
    tokenizer = BPETokenizer()
    tokenizer.train("low lower lowest", vocab_size=20)
    expected_ids = tokenizer.encode("lower")
    path = tmp_path / "tokenizer.json"
    tokenizer.save(path)
    loaded = BPETokenizer.load(path)
    assert loaded.encode("lower") == expected_ids
    assert loaded.decode(expected_ids) == "lower"
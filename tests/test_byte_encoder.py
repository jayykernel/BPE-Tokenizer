import pytest

from mini_bpe import ByteBPETokenizer


@pytest.fixture
def tokenizer():
    instance = ByteBPETokenizer()
    instance.train("hello hello world", vocab_size=300)
    return instance


def test_round_trip(tokenizer):
    text = "hello hello world"

    assert tokenizer.decode(tokenizer.encode(text)) == text


@pytest.mark.parametrize(
    "text",
    [
        "",
        "Hello",
        "Hello, world!",
        "你好",
        "😀",
        "café",
        "नमस्ते",
        "தமிழ்",
        "line one\nline two",
        "spaces   and tabs\t",
    ],
)
def test_unicode_round_trip(tokenizer, text):
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_unseen_characters(tokenizer):
    text = "🦄 New characters: こんにちは"

    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_empty_text(tokenizer):
    assert tokenizer.encode("") == []
    assert tokenizer.decode([]) == ""


def test_vocab_size_below_base_vocabulary():
    tokenizer = ByteBPETokenizer()

    with pytest.raises(ValueError):
        tokenizer.train("hello", vocab_size=100)


def test_untrained_tokenizer():
    tokenizer = ByteBPETokenizer()

    with pytest.raises(RuntimeError):
        tokenizer.encode("hello")


def test_save_load(tokenizer, tmp_path):
    text = "Hello 😀"

    expected = tokenizer.encode(text)
    path = tmp_path / "byte_tokenizer.json"

    tokenizer.save(path)

    loaded = ByteBPETokenizer.load(path)

    assert loaded.encode(text) == expected
    assert loaded.decode(expected) == text


def test_unknown_id(tokenizer):
    with pytest.raises(ValueError):
        tokenizer.decode([999999])

import pytest

from mini_bpe.decoder import decode, ids_to_tokens


@pytest.fixture
def vocab():
    return {
        "l": 0,
        "o": 1,
        "w": 2,
        "e": 3,
        "r": 4,
        "lo": 5,
        "low": 6,
    }


def test_ids_to_tokens(vocab):
    assert ids_to_tokens([6, 3, 4], vocab) == [
        "low",
        "e",
        "r",
    ]


def test_decode(vocab):
    assert decode([6, 3, 4], vocab) == "lower"


def test_decode_empty_ids(vocab):
    assert decode([], vocab) == ""


def test_decode_unknown_id(vocab):
    with pytest.raises(ValueError, match="Unknown token ID"):
        decode([999], vocab)

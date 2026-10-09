from mini_bpe.trainer import (
    count_pairs,
    get_best_pair,
    merge_pair,
    train_bpe,
)
def test_count_pairs():
    sequences = [
        ["l", "o", "w"],
        ["l", "o", "w"],
    ]
    counts = count_pairs(sequences)
    assert counts[("l", "o")] == 2
    assert counts[("o", "w")] == 2
def test_get_best_pair():
    counts = {
        ("l", "o"): 5,
        ("o", "w"): 3,
        ("w", "e"): 1,
    }
    assert get_best_pair(counts) == ("l", "o")
def test_merge_pair():
    sequence = ["l", "o", "w"]
    result = merge_pair(sequence, ("l", "o"))
    assert result == ["lo", "w"]
def test_train_bpe():
    sequences = [
        ["l", "o", "w"],
        ["l", "o", "w"],
        ["l", "o", "w"],
        ["l", "o", "w", "e", "r"],
    ]
    result, merges = train_bpe(sequences, num_merges=2)
    assert merges == [
        ("l", "o"),
        ("lo", "w"),
    ]
    assert result == [
        ["low"],
        ["low"],
        ["low"],
        ["low", "e", "r"],
    ]
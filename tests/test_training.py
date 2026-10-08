from mini_bpe.trainer import count_pairs, merge_pair
def test_count_pairs():
    sequences = [
        ["l", "o", "w"],
        ["l", "o", "w"],
    ]
    counts = count_pairs(sequences)
    assert counts[("l", "o")] == 2
    assert counts[("o", "w")] == 2
def test_merge_pair():
    sequence = ["l", "o", "w"]
    result = merge_pair(sequence, ("l", "o"))
    assert result == ["lo", "w"]
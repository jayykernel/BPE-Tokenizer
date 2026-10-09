from collections import Counter
Pair = tuple[str, str]
Sequence = list[str]
def count_pairs(sequences: list[Sequence]) -> Counter[Pair]:
    counts: Counter[Pair] = Counter()
    for sequence in sequences:
        for index in range(len(sequence) - 1):
            pair = (sequence[index], sequence[index + 1])
            counts[pair] += 1
    return counts
def get_best_pair(pair_counts: Counter[Pair]) -> Pair | None:
    if not pair_counts:
        return None
    return min(
        pair_counts,
        key=lambda pair: (
            -pair_counts[pair],
            pair[0],
            pair[1],
        ),
    )
def merge_pair(sequence: Sequence, pair: Pair) -> Sequence:
    first, second = pair
    merged: Sequence = []
    index = 0
    while index < len(sequence):
        if (
            index + 1 < len(sequence)
            and sequence[index] == first
            and sequence[index + 1] == second
        ):
            merged.append(first + second)
            index += 2
        else:
            merged.append(sequence[index])
            index += 1
    return merged
def train_bpe(
    sequences: list[Sequence],
    num_merges: int,
) -> tuple[list[Sequence], list[Pair]]:
    if num_merges < 0:
        raise ValueError("num_merges must be non-negative")
    sequences = [sequence.copy() for sequence in sequences]
    merges: list[Pair] = []
    for _ in range(num_merges):
        pair_counts = count_pairs(sequences)
        best_pair = get_best_pair(pair_counts)
        if best_pair is None:
            break
        merges.append(best_pair)
        sequences = [
            merge_pair(sequence, best_pair)
            for sequence in sequences
        ]
    return sequences, merges
def build_vocab(
    initial_symbols: set[str],
    merges: list[Pair],
) -> dict[str, int]:
    vocab = set(initial_symbols)
    for first, second in merges:
        vocab.add(first)
        vocab.add(second)
        vocab.add(first + second)
    return {
        token: token_id
        for token_id, token in enumerate(sorted(vocab))
    }
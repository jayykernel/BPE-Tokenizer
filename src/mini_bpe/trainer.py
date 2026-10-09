from collections import Counter
def count_pairs(sequences):
    counts = Counter()
    for sequence in sequences:
        for i in range(len(sequence) - 1):
            pair = (sequence[i], sequence[i + 1])
            counts[pair] += 1
    return counts
def get_best_pair(pair_counts):
    if not pair_counts:
        return None
    return max(pair_counts, key=pair_counts.get)
def merge_pair(sequence, pair):
    first, second = pair
    merged = []
    i = 0
    while i < len(sequence):
        if (
            i < len(sequence) - 1
            and sequence[i] == first
            and sequence[i + 1] == second
        ):
            merged.append(first + second)
            i += 2
        else:
            merged.append(sequence[i])
            i += 1
    return merged
def train_bpe(sequences, num_merges):
    merges = []
    for _ in range(num_merges):
        pair_counts = count_pairs(sequences)
        if not pair_counts:
            break
        best_pair = get_best_pair(pair_counts)
        merges.append(best_pair)
        sequences = [
            merge_pair(sequence, best_pair)
            for sequence in sequences
        ]
    return sequences, merges
sequences = [
    ["l", "o", "w"],
    ["l", "o", "w"],
    ["l", "o", "w"],
    ["l", "o", "w", "e", "r"],
]
sequences, merges = train_bpe(sequences, 2)
def build_vocab(sequences, merges):
    vocab = set()
    for sequence in sequences:
        vocab.update(sequence)
    for first, second in merges:
        vocab.add(first)
        vocab.add(second)
        vocab.add(first + second)
    sorted_tokens = sorted(vocab)
    return {
        token: token_id
        for token_id, token in enumerate(sorted_tokens)
    }
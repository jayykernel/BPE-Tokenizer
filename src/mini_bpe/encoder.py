from mini_bpe.trainer import merge_pair
def apply_merges(text, merges):
    sequence = list(text)
    for pair in merges:
        sequence = merge_pair(sequence, pair)
    return sequence
def tokens_to_ids(tokens, vocab):
    return [vocab[token] for token in tokens]
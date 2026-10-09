from mini_bpe.trainer import merge_pair
def apply_merges(text, merges):
    sequence = list(text)
    for pair in merges:
        sequence = merge_pair(sequence, pair)
    return sequence
def tokens_to_ids(tokens, vocab):
    return [vocab[token] for token in tokens] 
def encode(text, merges, vocab):
    tokens = apply_merges(text, merges)
    return tokens_to_ids(tokens, vocab)
from mini_bpe.encoder import encode
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
ids = encode("lower", merges, vocab)
print(ids)
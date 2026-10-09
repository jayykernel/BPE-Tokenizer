from mini_bpe.trainer import Pair, merge_pair


def apply_merges(text: str, merges: list[Pair]) -> list[str]:
    """Apply learned BPE merges to text in their training order."""
    tokens = list(text)

    for pair in merges:
        tokens = merge_pair(tokens, pair)

    return tokens


def tokens_to_ids(
    tokens: list[str],
    vocab: dict[str, int],
) -> list[int]:
    """Convert tokens to IDs, rejecting unknown tokens clearly."""
    ids: list[int] = []

    for token in tokens:
        if token not in vocab:
            raise ValueError(f"Token is not in the vocabulary: {token!r}")
        ids.append(vocab[token])

    return ids


def encode(
    text: str,
    merges: list[Pair],
    vocab: dict[str, int],
) -> list[int]:
    """Encode text into vocabulary IDs."""
    tokens = apply_merges(text, merges)
    return tokens_to_ids(tokens, vocab)

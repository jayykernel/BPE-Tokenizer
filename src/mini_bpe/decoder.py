def ids_to_tokens(
    ids: list[int],
    vocab: dict[str, int],
) -> list[str]:
    inverse_vocab = {token_id: token for token, token_id in vocab.items()}
    tokens = []
    for token_id in ids:
        if token_id not in inverse_vocab:
            raise ValueError(f"Unknown token ID: {token_id}")
        tokens.append(inverse_vocab[token_id])
    return tokens


def decode(
    ids: list[int],
    vocab: dict[str, int],
) -> str:
    """Reconstruct text from character-level token IDs."""
    return "".join(ids_to_tokens(ids, vocab))

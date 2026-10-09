import json
from pathlib import Path
from mini_bpe.decoder import decode as decode_ids
from mini_bpe.encoder import encode as encode_text
from mini_bpe.trainer import (
    Pair,
    build_vocab,
    train_bpe,
)
class BPETokenizer:
    """Educational character-level BPE tokenizer."""
    def __init__(self) -> None:
        self.vocab: dict[str, int] = {}
        self.merges: list[Pair] = []
        self.trained = False
    def train(self, corpus: str, vocab_size: int) -> None:
        if not corpus:
            raise ValueError("Training corpus must not be empty")
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        initial_symbols = set(corpus)
        if vocab_size < len(initial_symbols):
            raise ValueError(
                "vocab_size cannot be smaller than the number "
                "of unique initial characters"
            )
        sequences = [list(corpus)]
        num_merges = vocab_size - len(initial_symbols)
        _, merges = train_bpe(sequences, num_merges)
        self.vocab = build_vocab(initial_symbols, merges)
        self.merges = merges
        self.trained = True
    def encode(self, text: str) -> list[int]:
        self._check_trained()
        return encode_text(text, self.merges, self.vocab)
    def decode(self, ids: list[int]) -> str:
        self._check_trained()
        return decode_ids(ids, self.vocab)
    def save(self, path: str | Path) -> None:
        self._check_trained()
        data = {
            "format_version": 1,
            "algorithm": "character-level-bpe",
            "vocab": self.vocab,
            "merges": [list(pair) for pair in self.merges],
        }
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        source = Path(path)
        data = json.loads(source.read_text(encoding="utf-8"))
        if data.get("format_version") != 1:
            raise ValueError("Unsupported tokenizer format version")
        if data.get("algorithm") != "character-level-bpe":
            raise ValueError("Unsupported tokenizer algorithm")
        tokenizer = cls()
        tokenizer.vocab = {
            str(token): int(token_id)
            for token, token_id in data["vocab"].items()
        }
        tokenizer.merges = [
            (str(first), str(second))
            for first, second in data["merges"]
        ]
        tokenizer._validate_loaded_state()
        tokenizer.trained = True
        return tokenizer
    def _check_trained(self) -> None:
        if not self.trained:
            raise RuntimeError("Tokenizer has not been trained")
    def _validate_loaded_state(self) -> None:
        ids = list(self.vocab.values())
        if len(ids) != len(set(ids)):
            raise ValueError("Vocabulary contains duplicate IDs")
        if any(token_id < 0 for token_id in ids):
            raise ValueError("Vocabulary IDs must be non-negative")
        if set(ids) != set(range(len(ids))):
            raise ValueError("Vocabulary IDs must be contiguous")
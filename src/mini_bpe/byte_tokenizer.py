import json
from pathlib import Path

from mini_bpe.byte_trainer import (
    Pair,
    Sequence,
    train_bpe,
)


class ByteBPETokenizer:
    """Byte-level BPE tokenizer for arbitrary UTF-8 text."""

    def __init__(self) -> None:
        self.vocab: dict[bytes, int] = {}
        self.merges: list[Pair] = []
        self.trained = False

    def train(self, corpus: str, vocab_size: int = 300) -> None:
        if vocab_size < 256:
            raise ValueError("vocab_size must be at least 256")

        raw = corpus.encode("utf-8")

        sequences: list[Sequence] = [[bytes([byte]) for byte in raw]]

        initial_vocab = {bytes([byte]) for byte in range(256)}

        num_merges = vocab_size - 256
        _, merges = train_bpe(sequences, num_merges)

        vocab = set(initial_vocab)

        for first, second in merges:
            vocab.add(first)
            vocab.add(second)
            vocab.add(first + second)

        self.vocab = {token: token_id for token_id, token in enumerate(sorted(vocab))}

        self.merges = merges
        self.trained = True

    def encode(self, text: str) -> list[int]:
        self._check_trained()

        sequence = [bytes([byte]) for byte in text.encode("utf-8")]

        for pair in self.merges:
            sequence = self._merge(sequence, pair)

        return [self.vocab[token] for token in sequence]

    def decode(self, ids: list[int]) -> str:
        self._check_trained()

        inverse_vocab = {token_id: token for token, token_id in self.vocab.items()}

        try:
            raw = b"".join(inverse_vocab[token_id] for token_id in ids)
        except KeyError as exc:
            raise ValueError(f"Unknown token ID: {exc.args[0]}") from exc

        return raw.decode("utf-8")

    @staticmethod
    def _merge(
        sequence: Sequence,
        pair: Pair,
    ) -> Sequence:
        first, second = pair
        result: Sequence = []
        index = 0

        while index < len(sequence):
            if (
                index + 1 < len(sequence)
                and sequence[index] == first
                and sequence[index + 1] == second
            ):
                result.append(first + second)
                index += 2
            else:
                result.append(sequence[index])
                index += 1

        return result

    def save(self, path: str | Path) -> None:
        self._check_trained()

        data = {
            "format_version": 1,
            "algorithm": "byte-level-bpe",
            "vocab": [
                [token.hex(), token_id] for token, token_id in self.vocab.items()
            ],
            "merges": [[first.hex(), second.hex()] for first, second in self.merges],
        }

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)

        destination.write_text(
            json.dumps(data, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> ByteBPETokenizer:
        data = json.loads(Path(path).read_text(encoding="utf-8"))

        if data.get("format_version") != 1:
            raise ValueError("Unsupported format version")

        if data.get("algorithm") != "byte-level-bpe":
            raise ValueError("Unsupported tokenizer algorithm")

        tokenizer = cls()

        tokenizer.vocab = {
            bytes.fromhex(token): int(token_id) for token, token_id in data["vocab"]
        }

        tokenizer.merges = [
            (
                bytes.fromhex(first),
                bytes.fromhex(second),
            )
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

        if set(ids) != set(range(len(ids))):
            raise ValueError("Vocabulary IDs must be contiguous")

        expected_base = {bytes([byte]) for byte in range(256)}

        if not expected_base.issubset(self.vocab):
            raise ValueError("Missing base byte tokens")

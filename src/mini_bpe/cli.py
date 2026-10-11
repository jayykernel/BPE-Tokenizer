import argparse
from pathlib import Path

from mini_bpe import ByteBPETokenizer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and use a byte-level BPE tokenizer."
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--corpus", required=True)
    train_parser.add_argument("--vocab-size", type=int, default=1000)
    train_parser.add_argument("--output", default="tokenizer.json")

    encode_parser = subparsers.add_parser("encode")
    encode_parser.add_argument("--model", required=True)
    encode_parser.add_argument("--text", required=True)

    decode_parser = subparsers.add_parser("decode")
    decode_parser.add_argument("--model", required=True)
    decode_parser.add_argument("--ids", required=True)

    args = parser.parse_args()

    if args.command == "train":
        corpus = Path(args.corpus).read_text(encoding="utf-8")

        tokenizer = ByteBPETokenizer()
        tokenizer.train(corpus, vocab_size=args.vocab_size)
        tokenizer.save(args.output)

        print(f"Saved tokenizer to {args.output}")

    elif args.command == "encode":
        tokenizer = ByteBPETokenizer.load(args.model)
        print(tokenizer.encode(args.text))

    elif args.command == "decode":
        tokenizer = ByteBPETokenizer.load(args.model)
        ids = [int(value) for value in args.ids.split(",")]
        print(tokenizer.decode(ids))


if __name__ == "__main__":
    main()

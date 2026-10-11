# BPE-Tokenizer

A compact Python implementation of a byte-level BPE (Byte Pair Encoding) tokenizer for training, encoding, and decoding text efficiently. Designed for learning, experimentation, and integration into NLP workflows with minimal dependencies.

## Features

✨ **Byte-level tokenization** — Handle any text without a fixed Unicode vocabulary  
⚡ **Fast encoding/decoding** — Optimized for performance (see benchmarks below)  
🎓 **Easy to understand** — Clean, well-structured codebase for learning BPE internals  
🔧 **CLI included** — Train models, encode, and decode from the command line  
📦 **Zero dependencies** — No heavy libraries required  

## Why BPE-Tokenizer?

Byte-level BPE tokenizers are the foundation of modern language models like GPT. This project provides a lightweight, readable implementation perfect for:
- Understanding how subword tokenization works
- Benchmarking tokenizer performance
- Integrating custom tokenization into Python NLP pipelines
- Educational purposes and experimentation

## Installation

```bash
pip install -e .
```

Requires Python 3.14+

## Quick Start

### Train a tokenizer
```bash
bpe-tokenizer train --corpus ./data.txt --vocab-size 1000 --output tokenizer.json
```

### Encode text
```bash
bpe-tokenizer encode --model tokenizer.json --text "Hello, world!"
```

### Decode tokens
```bash
bpe-tokenizer decode --model tokenizer.json --ids "1,2,3,4,5"
```

## Benchmarks

Performance metrics from the standard benchmark run:

| Metric | Value |
|---|---:|
| **Corpus size** | 90,000 characters |
| **Token count** | 1 |
| **Vocabulary size** | 348 |
| **Training time** | 1.69 seconds |
| **Encoding time** | 0.68 seconds |
| **Decoding time** | 0.0004 seconds |

These benchmarks demonstrate the efficiency of the byte-level approach for both training and inference.

## Project Structure

```
├── src/mini_bpe/
│   ├── tokenizer.py         # Main BPE tokenizer class
│   ├── byte_tokenizer.py    # Byte-level variant
│   ├── trainer.py           # Training logic
│   ├── encoder.py           # Encoding utilities
│   ├── decoder.py           # Decoding utilities
│   └── cli.py               # Command-line interface
├── benchmarks/
│   └── benchmark.py         # Performance benchmarking scripts
├── tests/
│   ├── test_tokenizer.py
│   ├── test_byte_encoder.py
│   ├── test_encoder.py
│   ├── test_decoder.py
│   └── test_training.py
└── README.md
```

## Usage in Code

```python
from mini_bpe import ByteBPETokenizer

# Train a tokenizer
tokenizer = ByteBPETokenizer()
corpus = "your text data here..."
tokenizer.train(corpus, vocab_size=1000)

# Encode text
tokens = tokenizer.encode("Hello, world!")
print(tokens)  # [1, 2, 3, ...]

# Decode tokens
text = tokenizer.decode([1, 2, 3, ...])
print(text)  # "Hello, world!"

# Save and load
tokenizer.save("my_tokenizer.json")
loaded = ByteBPETokenizer.load("my_tokenizer.json")
```

## Development

### Running tests
```bash
pytest tests/
```

### Code quality
```bash
ruff check .
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Author

**Jayy** — [GitHub](https://github.com/jayykernel)

---

**Star this repo** ⭐ if you find it useful for learning or building with BPE tokenizers!

import time

from mini_bpe import ByteBPETokenizer

corpus = (
    "Byte Pair Encoding is a tokenization algorithm. "
    "It learns frequent adjacent token merges. "
) * 1000

start = time.perf_counter()

tokenizer = ByteBPETokenizer()
tokenizer.train(corpus, vocab_size=500)

training_seconds = time.perf_counter() - start

start = time.perf_counter()

ids = tokenizer.encode(corpus)

encoding_seconds = time.perf_counter() - start

start = time.perf_counter()

decoded = tokenizer.decode(ids)

decoding_seconds = time.perf_counter() - start

assert decoded == corpus

print(f"Corpus characters: {len(corpus)}")
print(f"Token count: {len(ids)}")
print(f"Vocabulary size: {len(tokenizer.vocab)}")
print(f"Training time: {training_seconds:.4f} seconds")
print(f"Encoding time: {encoding_seconds:.4f} seconds")
print(f"Decoding time: {decoding_seconds:.4f} seconds")

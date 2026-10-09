from collections import Counter

BytePair = tuple[int, int]
ByteSequence = list[int]


def count_byte_pairs(
    sequences: list[ByteSequence],
) -> Counter[BytePair]:
    counts: Counter[BytePair] = Counter()

    for sequence in sequences:
        for index in range(len(sequence) - 1):
            counts[(sequence[index], sequence[index + 1])] += 1

    return counts


def get_best_byte_pair(
    counts: Counter[BytePair],
) -> BytePair | None:
    if not counts:
        return None

    return min(
        counts,
        key=lambda pair: (
            -counts[pair],
            pair[0],
            pair[1],
        ),
    )


def merge_byte_pair(
    sequence: ByteSequence,
    pair: BytePair,
) -> ByteSequence:
    first, second = pair
    merged: ByteSequence = []
    index = 0

    while index < len(sequence):
        if (
            index + 1 < len(sequence)
            and sequence[index] == first
            and sequence[index + 1] == second
        ):
            # A merged token needs a distinct token identity.
            # Therefore, this simple list-of-integers function
            # cannot represent merged tokens as single integers.
            raise NotImplementedError("Use the token-sequence representation below")

        merged.append(sequence[index])
        index += 1

    return merged

"""Easy-to-read information-retrieval and duplicate-detection metrics."""

from collections.abc import Sequence


def precision_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """Fraction of the first K retrieved items that are relevant."""
    if k <= 0:
        raise ValueError("k must be greater than zero")
    top_k = retrieved[:k]
    return sum(item in relevant for item in top_k) / k


def recall_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """Fraction of all relevant items found in the first K results."""
    if not relevant:
        return 0.0
    return sum(item in relevant for item in retrieved[:k]) / len(relevant)


def duplicate_detection_accuracy(expected: Sequence[bool], predicted: Sequence[bool]) -> float:
    """Fraction of duplicate/non-duplicate decisions that were correct."""
    if len(expected) != len(predicted) or not expected:
        raise ValueError("Lists must have the same non-zero length")
    return sum(a == b for a, b in zip(expected, predicted)) / len(expected)

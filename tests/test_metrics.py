import pytest

from evaluation.metrics import duplicate_detection_accuracy, precision_at_k, recall_at_k


def test_retrieval_metrics():
    retrieved = ["cat.jpg", "car.jpg", "dog.jpg"]
    relevant = {"cat.jpg", "dog.jpg"}
    assert precision_at_k(retrieved, relevant, 3) == pytest.approx(2 / 3)
    assert recall_at_k(retrieved, relevant, 3) == 1.0


def test_duplicate_accuracy():
    assert duplicate_detection_accuracy([True, False, True], [True, False, False]) == pytest.approx(2 / 3)

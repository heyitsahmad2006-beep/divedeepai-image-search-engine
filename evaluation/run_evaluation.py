"""Evaluate labelled queries from evaluation/labels.json and save a chart."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from app.services.search_service import SearchService
from config.settings import BASE_DIR, settings
from evaluation.metrics import precision_at_k, recall_at_k


def main() -> None:
    labels_path = BASE_DIR / "evaluation" / "labels.json"
    if not labels_path.exists():
        raise SystemExit("Create evaluation/labels.json using labels.example.json as a guide.")
    labelled_queries = json.loads(labels_path.read_text(encoding="utf-8"))
    service = SearchService()
    if not service.index_ready:
        raise SystemExit("Build the index before running evaluation.")

    names, precisions, recalls = [], [], []
    for item in labelled_queries:
        results, _ = service.search(item["query"], item.get("k", 5))
        retrieved = [result["image_name"] for result in results]
        relevant = set(item["relevant_images"])
        names.append(item["query"])
        precisions.append(precision_at_k(retrieved, relevant, item.get("k", 5)))
        recalls.append(recall_at_k(retrieved, relevant, item.get("k", 5)))

    settings.evaluation_dir.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(names))
    width = 0.35
    plt.figure(figsize=(max(8, len(names) * 1.5), 5))
    plt.bar(x - width / 2, precisions, width, label="Precision@K")
    plt.bar(x + width / 2, recalls, width, label="Recall@K")
    plt.xticks(x, names, rotation=20, ha="right")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    output = settings.evaluation_dir / "retrieval_metrics.png"
    plt.savefig(output, dpi=150)
    print(f"Saved evaluation chart to {output}")


if __name__ == "__main__":
    main()

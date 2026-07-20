import time
import csv
import re
from pathlib import Path

import faiss
import numpy as np

from app.services.duplicate_service import DuplicateService
from app.utils.image_utils import category_from_path, find_image_files, validate_image
from config.settings import Settings, settings


class SearchService:
    """Coordinates image preparation, persistent FAISS indexing, and search."""

    def __init__(self, app_settings: Settings = settings, encoder=None) -> None:
        self.settings = app_settings
        self.encoder = encoder
        self.query_cache: dict[str, np.ndarray] = {}
        self.query_terms: list[str] = []
        self.query_vectors: np.ndarray | None = None
        self.duplicate_service = DuplicateService(app_settings.duplicate_hash_distance)
        self.index: faiss.Index | None = None
        self.metadata: list[dict[str, str]] = []
        self.search_latencies: list[float] = []
        self.load_saved_index()
        self.load_query_bank()

    def load_query_bank(self) -> None:
        """Load small precomputed CLIP text vectors without loading PyTorch."""
        if self.settings.query_bank_path.exists():
            bank = np.load(self.settings.query_bank_path)
            self.query_terms = bank["terms"].tolist()
            self.query_vectors = bank["vectors"].astype("float32")

    def encode_query_from_bank(self, query: str) -> np.ndarray:
        """Compose a CLIP-space query from matching precomputed concepts."""
        if self.query_vectors is None:
            raise RuntimeError("Query bank is missing. Run: python -m scripts.build_query_bank")
        cleaned = re.sub(r"[^a-z0-9 ]+", " ", query.lower())
        words = set(cleaned.split())
        matches = [i for i, term in enumerate(self.query_terms)
                   if term in words or (" " in term and term in cleaned)]
        if not matches:
            # Find the closest category spelling for unusual or plural words.
            matches = [i for i, term in enumerate(self.query_terms)
                       if any(word.startswith(term) or term.startswith(word) for word in words)]
        if not matches:
            raise ValueError("Try a more descriptive query using an object, color, scene, or category.")
        vector = self.query_vectors[matches].mean(axis=0, keepdims=True)
        vector /= np.linalg.norm(vector, axis=1, keepdims=True)
        return vector.astype("float32")

    def load_saved_index(self) -> None:
        if self.settings.index_path.exists() and self.settings.metadata_path.exists():
            self.index = faiss.read_index(str(self.settings.index_path))
            with self.settings.metadata_path.open(newline="", encoding="utf-8") as file:
                self.metadata = list(csv.DictReader(file))

    @property
    def index_ready(self) -> bool:
        return self.index is not None and self.index.ntotal > 0

    def build_index(self) -> dict[str, int | float]:
        from app.models.clip_model import ClipEncoder

        started = time.perf_counter()
        image_paths = [path for path in find_image_files(self.settings.dataset_dir) if validate_image(path)]
        if not image_paths:
            raise ValueError("No valid images found. Add images to data/dataset first.")

        unique_paths, hashes, duplicates = self.duplicate_service.remove_duplicates(image_paths)
        encoder = self.encoder or ClipEncoder(self.settings.model_name)
        embeddings = encoder.encode_images(unique_paths, self.settings.batch_size)
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)
        self.metadata = [
            {
                "image_path": str(path.resolve()),
                "image_name": path.name,
                "category": category_from_path(path, self.settings.dataset_dir),
            }
            for path in unique_paths
        ]

        np.save(self.settings.embeddings_path, embeddings)
        with self.settings.metadata_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=["image_path", "image_name", "category"])
            writer.writeheader()
            writer.writerows(self.metadata)
        faiss.write_index(self.index, str(self.settings.index_path))
        self.duplicate_service.save_hashes(hashes, self.settings.hashes_path)
        return {
            "images_indexed": len(unique_paths),
            "duplicates_skipped": duplicates,
            "embedding_dimension": embeddings.shape[1],
            "build_time_seconds": round(time.perf_counter() - started, 3),
        }

    def search(self, query: str, top_k: int) -> tuple[list[dict], float]:
        if not self.index_ready:
            raise RuntimeError("Index is not ready. Call POST /build-index first.")
        started = time.perf_counter()
        normalized_query = query.strip().lower()
        query_embedding = self.query_cache.get(normalized_query)
        if query_embedding is None:
            query_embedding = self.encode_query_from_bank(query)
            self.query_cache[normalized_query] = query_embedding
        actual_k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_embedding, actual_k)
        latency_ms = (time.perf_counter() - started) * 1000
        self.search_latencies.append(latency_ms)

        results = []
        for rank, (score, row_index) in enumerate(zip(scores[0], indices[0]), start=1):
            row = self.metadata[int(row_index)]
            results.append({
                "rank": rank,
                "image_name": row["image_name"],
                "image_url": f"/dataset/{Path(row['image_path']).relative_to(self.settings.dataset_dir).as_posix()}",
                "similarity_score": round(float(score), 4),
            })
        return results, round(latency_ms, 2)

    def metrics(self) -> dict:
        dimension = self.index.d if self.index is not None else 0
        last = self.search_latencies[-1] if self.search_latencies else 0.0
        average = float(np.mean(self.search_latencies)) if self.search_latencies else 0.0
        return {
            "index_ready": self.index_ready,
            "indexed_images": int(self.index.ntotal) if self.index is not None else 0,
            "embedding_dimension": int(dimension),
            "total_searches": len(self.search_latencies),
            "average_search_latency_ms": round(average, 2),
            "last_search_latency_ms": round(last, 2),
        }

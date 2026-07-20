from pathlib import Path

import imagehash
import joblib
from PIL import Image


class DuplicateService:
    """Detect exact and near duplicates with perceptual hashes."""

    def __init__(self, max_distance: int = 5) -> None:
        self.max_distance = max_distance

    @staticmethod
    def calculate_hash(path: Path) -> str:
        with Image.open(path) as image:
            return str(imagehash.phash(image.convert("RGB")))

    def find_duplicate(self, candidate_hash: str, known_hashes: dict[str, str]) -> str | None:
        candidate = imagehash.hex_to_hash(candidate_hash)
        for image_path, saved_hash in known_hashes.items():
            if candidate - imagehash.hex_to_hash(saved_hash) <= self.max_distance:
                return image_path
        return None

    def remove_duplicates(self, paths: list[Path]) -> tuple[list[Path], dict[str, str], int]:
        unique_paths: list[Path] = []
        hashes: dict[str, str] = {}
        # Integer XOR + bit_count is much faster than repeatedly constructing
        # ImageHash objects when the dataset contains thousands of files.
        known_hash_values: list[int] = []
        duplicate_count = 0
        for path in paths:
            image_hash = self.calculate_hash(path)
            hash_value = int(image_hash, 16)
            if any((hash_value ^ known).bit_count() <= self.max_distance for known in known_hash_values):
                duplicate_count += 1
                continue
            hashes[str(path)] = image_hash
            known_hash_values.append(hash_value)
            unique_paths.append(path)
        return unique_paths, hashes, duplicate_count

    @staticmethod
    def save_hashes(hashes: dict[str, str], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(hashes, path)

    @staticmethod
    def load_hashes(path: Path) -> dict[str, str]:
        return joblib.load(path) if path.exists() else {}

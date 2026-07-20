"""Download and arrange the free Caltech-256 dataset (about 1.2 GB)."""

import shutil
import tarfile
import urllib.request
from pathlib import Path

from config.settings import BASE_DIR, settings

URL = "https://data.caltech.edu/records/nyy15-4j048/files/256_ObjectCategories.tar?download=1"
MAX_IMAGES_PER_CATEGORY = 40


def main() -> None:
    settings.create_directories()
    archive = BASE_DIR / "data" / "caltech256.tar"
    extracted = BASE_DIR / "data" / "256_ObjectCategories"
    print("Downloading Caltech-256. This is a large download and can take a while...")
    urllib.request.urlretrieve(URL, archive)
    print("Extracting images...")
    with tarfile.open(archive) as file:
        file.extractall(BASE_DIR / "data", filter="data")
    copied = 0
    for category in sorted(extracted.iterdir()):
        if category.is_dir():
            destination = settings.dataset_dir / category.name
            destination.mkdir(parents=True, exist_ok=True)
            # Forty images from each of 256 categories gives about 10,000
            # diverse images while keeping laptop indexing practical.
            for image_path in sorted(category.iterdir())[:MAX_IMAGES_PER_CATEGORY]:
                if image_path.is_file():
                    target = destination / image_path.name
                    if not target.exists():
                        shutil.move(str(image_path), str(target))
                        copied += 1
    shutil.rmtree(extracted, ignore_errors=True)
    archive.unlink(missing_ok=True)
    print(f"Added {copied} Caltech images. Dataset is ready in {settings.dataset_dir}")


if __name__ == "__main__":
    main()

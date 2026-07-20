"""Download a small, diverse starter dataset for testing the complete app."""

import urllib.request
from pathlib import Path

from config.settings import settings


CATEGORIES = {
    "animals": "animal",
    "cars": "car",
    "nature": "mountain,nature",
    "buildings": "building,architecture",
    "food": "food",
    "people": "person,portrait",
    "technology": "computer,technology",
    "sports": "sports",
}


def main(images_per_category: int = 25) -> None:
    """Download deterministic Flickr sample images into category folders."""
    settings.create_directories()
    opener = urllib.request.build_opener()
    opener.addheaders = [("User-Agent", "Mozilla/5.0")]
    total = len(CATEGORIES) * images_per_category
    completed = 0

    for category_index, (category, keywords) in enumerate(CATEGORIES.items(), start=1):
        folder = settings.dataset_dir / category
        folder.mkdir(parents=True, exist_ok=True)
        for number in range(1, images_per_category + 1):
            destination = folder / f"{category}_varied_{number:03d}.jpg"
            if not destination.exists():
                # LoremFlickr requires a numeric lock for different deterministic images.
                lock_number = category_index * 1000 + number
                url = f"https://loremflickr.com/640/480/{keywords}?lock={lock_number}"
                try:
                    with opener.open(url, timeout=90) as response, destination.open("wb") as output:
                        output.write(response.read())
                except Exception as error:
                    destination.unlink(missing_ok=True)
                    print(f"Skipped {category} image {number}: {error}")
            completed += 1
            print(f"Progress: {completed}/{total}")

    print(f"Starter dataset is ready in {settings.dataset_dir}")


if __name__ == "__main__":
    main()

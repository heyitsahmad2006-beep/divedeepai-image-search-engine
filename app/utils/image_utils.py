from pathlib import Path

from PIL import Image, UnidentifiedImageError


ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def find_image_files(folder: Path) -> list[Path]:
    """Find supported images recursively and return a stable, sorted list."""
    return sorted(
        path for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in ALLOWED_EXTENSIONS
    )


def validate_image(path: Path) -> bool:
    """Use Pillow and OpenCV to reject corrupt or unreadable image files."""
    try:
        with Image.open(path) as image:
            image.verify()
        # OpenCV is imported lazily because its native library reserves a
        # large virtual-memory block on Windows. Searches do not need it.
        import cv2
        return cv2.imread(str(path)) is not None
    except (UnidentifiedImageError, OSError, MemoryError):
        return False


def open_rgb_image(path: Path) -> Image.Image:
    """CLIP expects three-channel RGB images."""
    with Image.open(path) as image:
        return image.convert("RGB").copy()


def category_from_path(path: Path, dataset_dir: Path) -> str:
    relative = path.relative_to(dataset_dir)
    return relative.parts[0] if len(relative.parts) > 1 else "uncategorized"

"""Build and save the complete CLIP/FAISS image index."""

from app.services.search_service import SearchService


def main() -> None:
    """Run the same indexing workflow used by POST /build-index."""
    result = SearchService().build_index()
    print(result, flush=True)


if __name__ == "__main__":
    main()

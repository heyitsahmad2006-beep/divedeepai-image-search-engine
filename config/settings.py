from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """All changeable application values live in one easy-to-find place."""

    app_name: str = "DiveDeepAI Project 1"
    model_name: str = "openai/clip-vit-base-patch32"
    dataset_dir: Path = BASE_DIR / "data" / "dataset"
    uploads_dir: Path = BASE_DIR / "data" / "uploads"
    embeddings_path: Path = BASE_DIR / "data" / "embeddings" / "image_embeddings.npy"
    metadata_path: Path = BASE_DIR / "data" / "embeddings" / "image_metadata.csv"
    hashes_path: Path = BASE_DIR / "data" / "embeddings" / "image_hashes.joblib"
    query_bank_path: Path = BASE_DIR / "data" / "embeddings" / "query_bank.npz"
    index_path: Path = BASE_DIR / "data" / "faiss_index" / "images.index"
    evaluation_dir: Path = BASE_DIR / "evaluation" / "results"
    static_dir: Path = BASE_DIR / "static"
    top_k_default: int = 5
    top_k_max: int = 50
    duplicate_hash_distance: int = 5
    batch_size: int = 4

    model_config = SettingsConfigDict(env_file=BASE_DIR / ".env", extra="ignore")

    def create_directories(self) -> None:
        for path in (
            self.dataset_dir,
            self.uploads_dir,
            self.embeddings_path.parent,
            self.index_path.parent,
            self.evaluation_dir,
            self.static_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


settings = Settings()

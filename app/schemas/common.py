"""Pydantic models for system and image endpoints."""

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    index_ready: bool


class ImageItem(BaseModel):
    name: str
    category: str
    url: str


class ImageListResponse(BaseModel):
    total: int
    skip: int
    limit: int
    images: list[ImageItem]


class UploadResponse(BaseModel):
    message: str
    filename: str | None
    saved_path: str | None
    is_duplicate: bool
    duplicate_of: str | None


class MetricsResponse(BaseModel):
    index_ready: bool
    indexed_images: int
    embedding_dimension: int
    total_searches: int
    average_search_latency_ms: float
    last_search_latency_ms: float

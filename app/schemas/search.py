from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=300, examples=["red sports car"])
    top_k: int = Field(default=5, ge=1, le=50)


class SearchResult(BaseModel):
    rank: int
    image_name: str
    image_url: str
    similarity_score: float


class SearchResponse(BaseModel):
    query: str
    top_k: int
    latency_ms: float
    results: list[SearchResult]


class BuildIndexResponse(BaseModel):
    message: str
    images_indexed: int
    duplicates_skipped: int
    embedding_dimension: int
    build_time_seconds: float


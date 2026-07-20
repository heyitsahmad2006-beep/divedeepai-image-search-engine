"""All HTTP endpoints for the image search application."""

from pathlib import Path
from shutil import copyfileobj
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse

from app.schemas.common import HealthResponse, ImageListResponse, MetricsResponse, UploadResponse
from app.schemas.search import BuildIndexResponse, SearchRequest, SearchResponse
from app.services.search_service import SearchService
from app.utils.image_utils import ALLOWED_EXTENSIONS, find_image_files, validate_image
from config.settings import settings

router = APIRouter()
search_service = SearchService()


@router.get("/health", response_model=HealthResponse, tags=["System"])
def health() -> dict:
    """Confirm that the API is running and report whether an index is ready."""
    return {"status": "healthy", "index_ready": search_service.index_ready}


@router.post("/build-index", response_model=BuildIndexResponse, tags=["Index"])
def build_index() -> dict:
    """Read dataset images, remove duplicates, create embeddings, and save the index."""
    try:
        result = search_service.build_index()
        return {"message": "Index built and saved successfully.", **result}
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Index build failed: {error}") from error


@router.post("/search", response_model=SearchResponse, tags=["Search"])
def search(request: SearchRequest) -> dict:
    """Turn text into a CLIP vector and return the nearest image vectors."""
    try:
        results, latency_ms = search_service.search(request.query.strip(), request.top_k)
        return {"query": request.query, "top_k": len(results), "latency_ms": latency_ms, "results": results}
    except RuntimeError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@router.post("/upload-image", response_model=UploadResponse, tags=["Images"])
def upload_image(file: UploadFile = File(...)) -> dict:
    """Validate an uploaded image and reject perceptual duplicates."""
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Use JPG, JPEG, PNG, WEBP, or BMP.")
    destination = settings.uploads_dir / f"{uuid4().hex}{suffix}"
    with destination.open("wb") as output:
        copyfileobj(file.file, output)
    if not validate_image(destination):
        destination.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid image.")

    candidate_hash = search_service.duplicate_service.calculate_hash(destination)
    saved_hashes = search_service.duplicate_service.load_hashes(settings.hashes_path)
    duplicate = search_service.duplicate_service.find_duplicate(candidate_hash, saved_hashes)
    if duplicate:
        destination.unlink(missing_ok=True)
        return {"message": "Duplicate image detected; file was not added.", "filename": file.filename,
                "saved_path": None, "is_duplicate": True, "duplicate_of": Path(duplicate).name}

    dataset_destination = settings.dataset_dir / "uploads" / destination.name
    dataset_destination.parent.mkdir(parents=True, exist_ok=True)
    destination.replace(dataset_destination)
    return {"message": "Image saved. Rebuild the index to make it searchable.", "filename": file.filename,
            "saved_path": str(dataset_destination), "is_duplicate": False, "duplicate_of": None}


@router.get("/images", response_model=ImageListResponse, tags=["Images"])
def list_images(skip: int = Query(0, ge=0), limit: int = Query(20, ge=1, le=100)) -> dict:
    """List dataset images with simple pagination."""
    paths = find_image_files(settings.dataset_dir)
    items = [{"name": p.name, "category": p.relative_to(settings.dataset_dir).parts[0]
              if len(p.relative_to(settings.dataset_dir).parts) > 1 else "uncategorized",
              "url": f"/dataset/{p.relative_to(settings.dataset_dir).as_posix()}"} for p in paths[skip:skip + limit]]
    return {"total": len(paths), "skip": skip, "limit": limit, "images": items}


@router.get("/metrics", response_model=MetricsResponse, tags=["System"])
def metrics() -> dict:
    """Return index size and measured search latency."""
    return search_service.metrics()


@router.get("/dataset/{image_path:path}", include_in_schema=False)
def dataset_image(image_path: str) -> FileResponse:
    """Safely serve a result image to the browser."""
    requested = (settings.dataset_dir / image_path).resolve()
    if settings.dataset_dir.resolve() not in requested.parents or not requested.is_file():
        raise HTTPException(status_code=404, detail="Image not found.")
    return FileResponse(requested)

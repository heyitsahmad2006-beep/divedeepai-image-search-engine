"""FastAPI application entry point. Run with: uvicorn app.main:app --reload"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from config.settings import settings


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings.create_directories()
    yield


app = FastAPI(
    title=settings.app_name,
    description="Beginner-friendly semantic image retrieval with pretrained CLIP and FAISS.",
    version="1.0.0",
    lifespan=lifespan,
)
app.include_router(router)
app.mount("/static", StaticFiles(directory=settings.static_dir), name="static")


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    return FileResponse(settings.static_dir / "index.html")

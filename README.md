# DiveDeepAI Project 1

A beginner-friendly visual image search engine built with FastAPI, pretrained OpenAI CLIP, and FAISS. Users enter a text query such as `red sports car` or `snowy mountains`, and the application returns the most semantically similar indexed images with ranks and similarity scores.

This is an **image-retrieval** project, not an image-classification project. It uses pretrained models and does not train a neural network from scratch.

## Features

- Text-to-image semantic search
- Pretrained `openai/clip-vit-base-patch32`
- Exact FAISS similarity search
- Configurable Top-K results from 1 to 50
- Persistent embeddings and index files
- Perceptual duplicate detection with ImageHash
- Image validation with Pillow and OpenCV
- Image upload and paginated image listing APIs
- Precision@K, Recall@K, duplicate accuracy, and latency metrics
- Responsive light/dark web interface
- Swagger API documentation and Pytest tests

The current local index contains **10,463 unique images**, 512-dimensional embeddings, and a bank of 547 CLIP text concepts.

## How it works

### Image indexing

1. Images are found recursively in `data/dataset`.
2. Pillow and OpenCV reject unreadable files.
3. ImageHash removes duplicate and near-duplicate images.
4. CLIP creates a normalized 512-dimensional vector for every unique image.
5. NumPy, CSV, Joblib, and FAISS save the embeddings, metadata, hashes, and search index locally.

### Text search

Live searches use `data/embeddings/query_bank.npz`, which contains CLIP vectors for dataset category names and common concepts. Matching concept vectors are averaged and normalized, then FAISS compares the query vector with every indexed image vector.

This lightweight design avoids loading PyTorch during each API request. Queries should include represented objects, categories, colors, scenes, or activities. Rebuild the query bank after changing category folders or search concepts.

## Project structure

```text
ai-image-search-engine/
├── app/
│   ├── api/routes.py              # FastAPI endpoints
│   ├── models/clip_model.py       # CLIP image/text encoder
│   ├── schemas/                   # Pydantic models
│   ├── services/                  # Search and duplicate logic
│   ├── utils/image_utils.py       # Image validation utilities
│   └── main.py                    # FastAPI entry point
├── config/settings.py             # Paths and configuration
├── data/
│   ├── dataset/                   # Dataset images
│   ├── embeddings/                # Generated vectors and metadata
│   ├── faiss_index/               # Generated FAISS index
│   └── uploads/                   # Uploaded images
├── evaluation/                    # Metrics and evaluation script
├── scripts/                       # Dataset and index commands
├── static/                        # HTML, CSS, and JavaScript UI
├── tests/                         # Automated tests
├── requirements.txt
└── README.md
```

## Installation

Open the project folder in VS Code:

```text
C:\Users\hp\Desktop\ai-image-search-engine
```

Create and activate a Python 3.11 virtual environment:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt --default-timeout 1000
```

## Dataset

Use the existing images, add your own images under category subfolders in `data/dataset`, or download a dataset.

Small eight-category starter dataset:

```powershell
python -m scripts.download_starter_dataset
```

Approximately 10,000 images selected from Caltech-256:

```powershell
python -m scripts.download_caltech256
```

The Caltech download is approximately 1.2 GB. The script keeps up to 40 images from each of 256 categories.

## Build the search artifacts

After changing dataset images, rebuild the image index:

```powershell
python -m scripts.build_index
```

After changing category folders or `COMMON_TERMS` in `scripts/build_query_bank.py`, rebuild the query bank:

```powershell
python -m scripts.build_query_bank
```

For a new dataset, run both commands in that order. Large CPU index builds can take a long time. Avoid running other memory-heavy Python services during indexing.

## Run the application

```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8001
```

Open:

- Web interface: `http://127.0.0.1:8001`
- Swagger UI: `http://127.0.0.1:8001/docs`
- Health check: `http://127.0.0.1:8001/health`

Stop the server with `Ctrl+C`.

## API endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | API and index status |
| `POST` | `/search` | Search images using text |
| `POST` | `/build-index` | Build and save the image index |
| `POST` | `/upload-image` | Validate and add a non-duplicate image |
| `GET` | `/images` | List dataset images with pagination |
| `GET` | `/metrics` | Index size and search latency |
| `GET` | `/dataset/{path}` | Serve a result image safely |

Example search request:

```json
{
  "query": "red sports car",
  "top_k": 10
}
```

Example result item:

```json
{
  "rank": 1,
  "image_name": "example.jpg",
  "image_url": "/dataset/cars/example.jpg",
  "similarity_score": 0.286
}
```

Uploaded images are stored under `data/dataset/uploads` but are not searchable until the image index is rebuilt and the API is restarted.

## Evaluation and tests

Run automated tests:

```powershell
python -m pytest -q
```

For retrieval evaluation, copy and edit the label example:

```powershell
Copy-Item evaluation\labels.example.json evaluation\labels.json
python -m evaluation.run_evaluation
```

The evaluation script calculates Precision@K and Recall@K and saves a chart to `evaluation/results/retrieval_metrics.png`. Duplicate accuracy is implemented in `evaluation/metrics.py`. Search latency is returned by `/search` and summarized by `/metrics`.

## Generated files

```text
data/embeddings/image_embeddings.npy
data/embeddings/image_metadata.csv
data/embeddings/image_hashes.joblib
data/embeddings/query_bank.npz
data/faiss_index/images.index
```

These files, the dataset, `.venv`, `.env`, and Python caches are excluded from GitHub by `.gitignore`. Anyone cloning the repository must download/add a dataset and build the index and query bank locally.

## Configuration

Defaults are stored in `config/settings.py` and can be overridden with a local `.env` file.

| Setting | Default |
|---|---|
| App name | `DiveDeepAI Project 1` |
| CLIP model | `openai/clip-vit-base-patch32` |
| Maximum Top-K | `50` |
| Duplicate hash distance | `5` |
| Image batch size | `4` |

## Important notes

- Similarity scores are ranking values, not probability percentages.
- The project uses pretrained CLIP; it does not train CLIP.
- Restart the API after rebuilding index artifacts.
- Rebuild the image index after adding, deleting, or replacing images.
- Rebuild the query bank after changing categories or supported concepts.

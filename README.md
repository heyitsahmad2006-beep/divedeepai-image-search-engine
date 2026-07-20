# DiveDeepAI Project 1

DiveDeepAI Project 1 is a text-to-image search application built with FastAPI, pretrained OpenAI CLIP embeddings, and FAISS similarity search. A user describes an image in natural language, chooses how many results to return, and receives the closest indexed images with similarity scores.

The project performs **image retrieval**, not image classification. It does not assign one fixed label to an image. Instead, it represents images and supported text concepts as normalized 512-dimensional CLIP vectors and compares those vectors using a FAISS inner-product index.

The repository also includes perceptual-hash duplicate detection, persistent local artifacts, image uploads, evaluation metrics, automated tests, API documentation, and a responsive light/dark web interface.

## Current verified state

The local project currently contains a completed index with:

- 10,463 unique indexed images
- 512 values per CLIP embedding
- 57 duplicate or near-duplicate images excluded during preparation
- 547 precomputed CLIP query concepts
- A saved FAISS index and metadata that load when the API starts

The exact image count can always be checked through `GET /metrics`.

## How search works

### Offline indexing

1. Images are discovered recursively inside `data/dataset`.
2. Pillow and OpenCV validate that each file is readable.
3. ImageHash calculates a perceptual hash for each image.
4. Images whose perceptual-hash Hamming distance is at most `5` are treated as duplicates or near duplicates.
5. The pretrained `openai/clip-vit-base-patch32` vision encoder generates a 512-dimensional vector for every unique image.
6. Scikit-learn L2-normalizes the vectors.
7. NumPy saves the embeddings, CSV stores image metadata, Joblib stores perceptual hashes, and FAISS saves the similarity index.

### Live text search

The live API is intentionally lightweight and does not load PyTorch for every request.

1. `scripts/build_query_bank.py` uses CLIP's text encoder offline to create vectors for dataset category names and common concepts such as colors, objects, scenes, sports, food, people, and technology.
2. These terms and vectors are saved in `data/embeddings/query_bank.npz`.
3. A submitted query is cleaned and matched against the saved concepts.
4. Vectors for the matched concepts are averaged and normalized to create the query vector.
5. FAISS compares that vector with all saved image vectors.
6. The API returns the Top-K closest images, ranks, URLs, similarity scores, and measured latency.
7. Composed query vectors are cached in memory for repeated searches during the current server session.

This design keeps live search stable on memory-limited Windows machines. A query must contain a concept represented by the query bank. If dataset categories are changed, rebuild the query bank.

## Technologies and installed dependencies

- Python 3.11+
- FastAPI and Uvicorn
- PyTorch CPU
- Hugging Face Transformers
- OpenAI CLIP ViT-B/32
- FAISS CPU
- NumPy
- Pandas is included in `requirements.txt`; the current search service reads and writes metadata with Python's built-in `csv` module.
- Pillow and OpenCV
- Scikit-learn
- ImageHash and Joblib
- Matplotlib
- Pydantic and pydantic-settings
- HTML, CSS, and JavaScript
- Pytest and HTTPX

## Project structure

```text
ai-image-search-engine/
├── app/
│   ├── api/
│   │   └── routes.py              # All FastAPI endpoints
│   ├── models/
│   │   └── clip_model.py          # CLIP image and text encoder wrapper
│   ├── schemas/
│   │   ├── common.py              # Health, image, upload, and metric models
│   │   └── search.py              # Search and index response models
│   ├── services/
│   │   ├── duplicate_service.py   # Perceptual-hash duplicate detection
│   │   └── search_service.py      # Persistent index loading and search logic
│   ├── utils/
│   │   └── image_utils.py         # Discovery, validation, RGB conversion
│   └── main.py                    # FastAPI application entry point
├── config/
│   └── settings.py                # Paths and configurable values
├── data/
│   ├── dataset/                   # Searchable images arranged in folders
│   ├── embeddings/                # Embeddings, metadata, hashes, query bank
│   ├── faiss_index/               # Saved FAISS index
│   └── uploads/                   # Temporary upload location
├── evaluation/
│   ├── labels.example.json        # Relevance-label example
│   ├── metrics.py                 # Precision@K, Recall@K, duplicate accuracy
│   └── run_evaluation.py          # Evaluation runner and chart generator
├── scripts/
│   ├── build_index.py             # Command-line image-index builder
│   ├── build_query_bank.py        # Command-line CLIP concept-bank builder
│   ├── download_caltech256.py     # Approximately 10K-image dataset setup
│   └── download_starter_dataset.py # Small eight-category starter dataset
├── static/
│   ├── app.js                     # Search requests, theme, result rendering
│   ├── index.html                 # Browser interface
│   └── style.css                  # Responsive light/dark design
├── tests/
│   ├── test_api.py                # Health endpoint test
│   └── test_metrics.py            # Evaluation formula tests
├── .env.example                   # Optional environment overrides
├── .gitignore                     # Generated/local files excluded from Git
├── requirements.txt               # Python dependencies
└── README.md
```

Folders named `__pycache__` and `.pytest_cache` are generated automatically by Python and Pytest. They are not application modules.

## Important files and functions

### `app/main.py`

Creates the FastAPI application, creates required directories during startup, registers the API router, serves static assets under `/static`, and returns the browser interface from `/`.

### `app/services/search_service.py`

`SearchService` is the central service.

- `load_saved_index()` loads the saved FAISS index and CSV metadata.
- `load_query_bank()` loads saved CLIP concepts and vectors.
- `encode_query_from_bank()` cleans a query, finds supported concepts, averages their CLIP vectors, and normalizes the result.
- `build_index()` validates images, removes duplicates, generates image embeddings, and saves every index artifact.
- `search()` searches FAISS and constructs ranked API results.
- `metrics()` reports index size and in-memory latency statistics.

### `app/models/clip_model.py`

`ClipEncoder` wraps `openai/clip-vit-base-patch32`.

- `load_image_model()` loads the full CLIP model for offline image indexing.
- `load_text_model()` loads only CLIP's text tower for offline concept-bank generation.
- `encode_images()` creates normalized image vectors in batches.
- `encode_text()` creates a normalized vector for one text string.

The class uses CUDA if PyTorch reports an available GPU; otherwise it uses the CPU. PyTorch CPU threads are limited to one for stability on the Windows machine for which this project was prepared.

### `app/services/duplicate_service.py`

Calculates perceptual hashes, compares hashes using XOR and integer bit counts, skips near duplicates during indexing, and stores accepted hashes with Joblib. Uploaded images are also compared against saved hashes.

### `app/utils/image_utils.py`

- `find_image_files()` recursively finds JPG, JPEG, PNG, WEBP, and BMP files.
- `validate_image()` verifies each image with Pillow and then checks it with OpenCV.
- `open_rgb_image()` converts an image to the RGB format expected by CLIP.
- `category_from_path()` uses the first dataset subfolder as the category.

OpenCV is imported only inside `validate_image()` so normal searches do not reserve OpenCV's native memory.

## Saved artifacts

The application uses these generated files:

| File | Purpose |
|---|---|
| `data/embeddings/image_embeddings.npy` | Normalized CLIP image vectors |
| `data/embeddings/image_metadata.csv` | Image paths, names, and categories in FAISS row order |
| `data/embeddings/image_hashes.joblib` | Perceptual hashes for duplicate checking |
| `data/embeddings/query_bank.npz` | Supported text concepts and CLIP text vectors |
| `data/faiss_index/images.index` | Persistent FAISS inner-product index |

The metadata row order must remain aligned with the FAISS vector order. Rebuild the index after changing the dataset instead of editing generated artifacts manually.

## Dataset options

### Existing dataset

If `data/dataset` and the generated artifacts are already present, no dataset download or index build is required. Start the API directly.

### Small starter dataset

The starter downloader requests 25 images for each of eight categories: animals, cars, nature, buildings, food, people, technology, and sports.

```powershell
python -m scripts.download_starter_dataset
```

### Approximately 10,000 Caltech-256 images

The large downloader retrieves the approximately 1.2 GB Caltech-256 archive and keeps at most 40 files from every category. With 256 categories, this adds approximately 10,000 images.

```powershell
python -m scripts.download_caltech256
```

The script extracts into `data`, moves selected images into category folders under `data/dataset`, and removes the downloaded archive and temporary extraction directory afterward.

## Installation in VS Code on Windows

Open `C:\Users\hp\Desktop\ai-image-search-engine` as the VS Code folder, then open **Terminal > New Terminal**.

### 1. Create the virtual environment

```powershell
py -3.11 -m venv .venv
```

### 2. Activate it

```powershell
.\.venv\Scripts\Activate.ps1
```

If local script execution is disabled:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt --default-timeout 1000
```

The requirements select the CPU-only PyTorch 2.6 wheel and NumPy/OpenCV versions used by this code.

## Running the current application

Activate the virtual environment and start Uvicorn on port 8001:

```powershell
.\.venv\Scripts\Activate.ps1
python -m uvicorn app.main:app --host 127.0.0.1 --port 8001
```

Open:

- Web interface: `http://127.0.0.1:8001`
- Swagger API documentation: `http://127.0.0.1:8001/docs`
- OpenAPI JSON: `http://127.0.0.1:8001/openapi.json`

Stop the server with `Ctrl+C`.

The browser interface includes:

- Natural-language query input
- Configurable Top-K from 1 to 50
- Suggested searches
- Ranked image cards and similarity scores
- Search latency display
- Persistent light/dark theme preference stored in browser local storage
- Link to Swagger API documentation

## Building or rebuilding artifacts

### Build the image index

Run this after adding, removing, or replacing dataset images:

```powershell
python -m scripts.build_index
```

The same operation is exposed through `POST /build-index`, but the command-line script is easier to monitor for a large CPU build. Indexing loads the full CLIP vision model and can take a long time for thousands of images.

### Build the query concept bank

Run this after adding or renaming dataset category folders, or after changing `COMMON_TERMS` in `scripts/build_query_bank.py`:

```powershell
python -m scripts.build_query_bank
```

Start or restart the API afterward so `SearchService` reloads the new bank.

### Recommended order for a new dataset

```powershell
python -m scripts.build_index
python -m scripts.build_query_bank
python -m uvicorn app.main:app --host 127.0.0.1 --port 8001
```

Do not run a large index build while the API or another memory-heavy PyTorch service is active on a machine with limited virtual memory.

## API endpoints

### `GET /health`

Confirms that the API process is running and reports whether FAISS has a non-empty index.

Example response:

```json
{
  "status": "healthy",
  "index_ready": true
}
```

### `POST /search`

Searches the saved image index.

Request:

```json
{
  "query": "red sports car",
  "top_k": 10
}
```

Constraints:

- `query`: 1 to 300 characters
- `top_k`: 1 to 50

Response fields:

- `query`: submitted query
- `top_k`: actual number returned
- `latency_ms`: query composition and FAISS search time
- `results`: ranked images
- `similarity_score`: FAISS inner-product score between normalized vectors

### `POST /build-index`

Runs the complete image-index build inside the API process. It returns the number indexed, duplicates skipped, embedding dimension, and total build time. It returns HTTP 400 if no valid images exist and HTTP 500 for build failures.

### `POST /upload-image`

Accepts one multipart form-data field named `file`.

- Supported extensions: JPG, JPEG, PNG, WEBP, BMP
- Invalid images are deleted and rejected with HTTP 400.
- Duplicate images are deleted and reported with `is_duplicate: true`.
- New images are moved into `data/dataset/uploads`.
- A successful upload is not searchable until the image index is rebuilt.

### `GET /images`

Lists dataset files with pagination.

Query parameters:

- `skip`: default `0`, minimum `0`
- `limit`: default `20`, range `1` to `100`

### `GET /metrics`

Returns:

- `index_ready`
- `indexed_images`
- `embedding_dimension`
- `total_searches`
- `average_search_latency_ms`
- `last_search_latency_ms`

Latency values and search count are kept in memory and reset when the API restarts.

### `GET /dataset/{image_path}`

Serves result images from `data/dataset`. The route resolves the requested path and rejects paths outside the dataset directory.

## Testing with Swagger UI

1. Start the server.
2. Open `http://127.0.0.1:8001/docs`.
3. Run `GET /health` and confirm `index_ready` is `true`.
4. Run `GET /metrics` and confirm `indexed_images` is non-zero.
5. Run `POST /search` with a query that uses represented concepts.
6. Run `GET /images` to inspect paginated dataset entries.
7. Test `POST /upload-image` with multipart form-data.

## Automated tests

Run:

```powershell
python -m pytest -q
```

The current tests verify:

- The health endpoint responds successfully.
- Precision@K returns the expected value.
- Recall@K returns the expected value.
- Duplicate-detection accuracy returns the expected value.

The test suite does not rebuild the CLIP model or the 10K-image index.

## Evaluation

### Precision@K

The number of relevant images among the first K results divided by K.

### Recall@K

The number of relevant images retrieved in the first K divided by the total number of known relevant images.

### Duplicate-detection accuracy

The fraction of labelled duplicate/non-duplicate decisions predicted correctly.

### Search latency

Elapsed time for query-vector composition and FAISS search. It is recorded by `SearchService.search()` and exposed through `/search` and `/metrics`.

### Run the labelled evaluation

Copy `evaluation/labels.example.json` to `evaluation/labels.json`, then replace the example filenames with actual relevant filenames from this dataset.

```powershell
Copy-Item evaluation\labels.example.json evaluation\labels.json
python -m evaluation.run_evaluation
```

The script searches every labelled query, calculates Precision@K and Recall@K, and saves:

```text
evaluation/results/retrieval_metrics.png
```

## Configuration

Defaults are defined in `config/settings.py`. They can be overridden by creating `.env` from `.env.example`.

| Setting | Current default |
|---|---|
| `APP_NAME` | `DiveDeepAI Project 1` |
| `MODEL_NAME` | `openai/clip-vit-base-patch32` |
| `TOP_K_DEFAULT` | `5` |
| `TOP_K_MAX` | `50` |
| `DUPLICATE_HASH_DISTANCE` | `5` |
| `BATCH_SIZE` | `4` |

All data and artifact paths are derived from the project root by `config/settings.py`.

## Error behavior and operational notes

- Search returns a conflict response if no FAISS index has been loaded.
- Search requires `query_bank.npz`; its absence produces a message instructing you to run `python -m scripts.build_query_bank`.
- Queries without a represented concept cannot be composed. Use a more descriptive object, category, color, scene, or activity, or add the required term to `COMMON_TERMS` and rebuild the bank.
- Similarity scores are CLIP/FAISS ranking values, not probabilities or confidence percentages.
- Adding an upload does not modify the in-memory index automatically.
- Rebuilding artifacts while the API is running does not reload the `SearchService` instance. Restart the API afterward.
- The application does not train CLIP and does not train a custom neural network.
- The Hugging Face model is downloaded on first use and then reused from the local Hugging Face cache.

## Interview explanation

A concise explanation of the system is:

> DiveDeepAI Project 1 is an image-retrieval system based on pretrained CLIP and FAISS. During offline indexing, it validates images, removes perceptual duplicates, creates normalized CLIP image vectors, and stores them in an exact FAISS inner-product index. It also creates an offline bank of CLIP text vectors for dataset categories and common search concepts. At runtime, the FastAPI service composes a normalized query vector from that bank, searches FAISS, and returns ranked image URLs and similarity scores. Persistent artifacts prevent embeddings from being regenerated whenever the server starts.

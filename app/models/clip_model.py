from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import normalize
from transformers import AutoTokenizer, CLIPModel, CLIPProcessor, CLIPTextModelWithProjection

from app.utils.image_utils import open_rgb_image
from config.settings import settings


class ClipEncoder:
    """Small wrapper around pretrained CLIP image and text encoders."""

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or settings.model_name
        # A single CPU thread is more stable on memory-limited Windows laptops.
        torch.set_num_threads(1)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.image_model: CLIPModel | None = None
        self.image_processor: CLIPProcessor | None = None
        self.text_model: CLIPTextModelWithProjection | None = None
        self.tokenizer = None

    def load_image_model(self) -> None:
        """Load full CLIP only for offline image-index building."""
        if self.image_model is None:
            self.image_processor = CLIPProcessor.from_pretrained(self.model_name)
            self.image_model = CLIPModel.from_pretrained(self.model_name).to(self.device)
            self.image_model.eval()

    def load_text_model(self) -> None:
        """Load only CLIP's smaller text tower for API searches."""
        if self.text_model is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.text_model = CLIPTextModelWithProjection.from_pretrained(self.model_name).to(self.device)
            self.text_model.eval()

    def encode_images(self, image_paths: list[Path], batch_size: int = 16) -> np.ndarray:
        self.load_image_model()
        all_embeddings: list[np.ndarray] = []
        assert self.image_processor is not None and self.image_model is not None

        for start in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[start:start + batch_size]
            images = [open_rgb_image(path) for path in batch_paths]
            inputs = self.image_processor(images=images, return_tensors="pt", padding=True)
            inputs = {name: value.to(self.device) for name, value in inputs.items()}
            with torch.inference_mode():
                features = self.image_model.get_image_features(**inputs)
            all_embeddings.append(features.cpu().numpy())

        embeddings = np.vstack(all_embeddings).astype("float32")
        return normalize(embeddings, norm="l2").astype("float32")

    def encode_text(self, text: str) -> np.ndarray:
        self.load_text_model()
        assert self.tokenizer is not None and self.text_model is not None
        inputs = self.tokenizer(text=[text], return_tensors="pt", padding=True, truncation=True)
        inputs = {name: value.to(self.device) for name, value in inputs.items()}
        with torch.inference_mode():
            features = self.text_model(**inputs).text_embeds
        return normalize(features.cpu().numpy(), norm="l2").astype("float32")

"""Precompute reusable CLIP text vectors so API searches stay lightweight."""

import re

import numpy as np
import torch

from app.models.clip_model import ClipEncoder
from config.settings import settings

COMMON_TERMS = """
red blue green yellow black white orange purple pink brown gray silver gold colorful
car automobile vehicle sports racing classic vintage modern fast road motorcycle bicycle
animal dog puppy cat kitten bird bear horse fish insect wildlife cute furry
mountain snowy snow lake river ocean beach forest tree flower desert sunset sunrise nature landscape
building architecture house castle tower bridge city street church temple interior exterior
food meal dish fruit vegetable cake dessert pizza bread drink coffee restaurant delicious
people person man woman child baby family portrait face crowd happy smiling professional
technology computer laptop phone camera robot machine electronics science futuristic
sport athlete football soccer basketball baseball tennis swimming running cycling action
indoor outdoor closeup macro aerial night day bright dark beautiful dramatic peaceful
""".split()


def main() -> None:
    category_phrases = []
    for folder in settings.dataset_dir.iterdir():
        if folder.is_dir():
            clean = re.sub(r"^\d+[.\-_]?", "", folder.name)
            clean = clean.replace("-", " ").replace("_", " ").strip().lower()
            if clean:
                category_phrases.append(clean)
    terms = sorted(set(COMMON_TERMS + category_phrases + [word for phrase in category_phrases for word in phrase.split()]))

    encoder = ClipEncoder()
    encoder.load_text_model()
    assert encoder.tokenizer is not None and encoder.text_model is not None
    inputs = encoder.tokenizer(terms, return_tensors="pt", padding=True, truncation=True)
    inputs = {name: value.to(encoder.device) for name, value in inputs.items()}
    with torch.inference_mode():
        vectors = encoder.text_model(**inputs).text_embeds.cpu().numpy().astype("float32")
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    np.savez_compressed(settings.query_bank_path, terms=np.asarray(terms), vectors=vectors)
    print(f"Saved {len(terms)} CLIP query concepts to {settings.query_bank_path}")


if __name__ == "__main__":
    main()

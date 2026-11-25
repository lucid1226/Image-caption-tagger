from pathlib import Path
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

_model = None
_processor = None

DEFAULT_CANDIDATE_TAGS = [
    "cat", "dog", "pet", "animal", "kitten", "puppy",
    "person", "people", "portrait", "smile",
    "nature", "forest", "grass", "sky", "beach", "mountain", "flower",
    "food", "coffee", "dessert", "burger", "pizza",
    "car", "street", "city", "building", "indoor", "outdoor",
    "laptop", "book", "art", "sports", "travel"
]

def extract_tags(
    image_path: str | Path,
    candidate_tags: list[str] | None = None,
    top_k: int = 5,
    min_prob: float = 0.10,
) -> list[str]:
    global _model, _processor
    if _model is None or _processor is None:
        model_name = "openai/clip-vit-base-patch32"
        _model = CLIPModel.from_pretrained(model_name)
        _processor = CLIPProcessor.from_pretrained(model_name)
        _model.eval()

    tags = candidate_tags or DEFAULT_CANDIDATE_TAGS
    prompts = [f"a photo of {tag}" for tag in tags]

    image = Image.open(image_path).convert("RGB")
    inputs = _processor(text=prompts, images=image, return_tensors="pt", padding=True)

    with torch.no_grad():
        outputs = _model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)[0]

    top_indices = torch.argsort(probs, descending=True)[:top_k]
    selected = [
        f"#{tags[idx]}"
        for idx in top_indices
        if float(probs[idx]) >= min_prob
    ]

    if not selected:
        best_idx = int(top_indices[0])
        selected = [f"#{tags[best_idx]}"]

    return selected
from pathlib import Path
import numpy as np
from PIL import Image

def preprocess_image(image_path: str | Path, size: tuple[int, int] = (416, 416)) -> np.ndarray:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(str(image_path))

    img = Image.open(path).convert("RGB").resize(size)
    return np.asarray(img, dtype=np.float32) / 255.0
from pathlib import Path
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

_processor = None
_model = None

def generate_caption(image_path: str | Path) -> str:
    global _processor, _model
    if _processor is None or _model is None:
        model_name = "Salesforce/blip-image-captioning-base"
        _processor = BlipProcessor.from_pretrained(model_name)
        _model = BlipForConditionalGeneration.from_pretrained(model_name)

    image = Image.open(image_path).convert("RGB")
    inputs = _processor(images=image, return_tensors="pt")
    output_ids = _model.generate(**inputs, max_new_tokens=30)
    caption = _processor.decode(output_ids[0], skip_special_tokens=True)
    return caption.strip()
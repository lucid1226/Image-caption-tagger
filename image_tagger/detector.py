import cv2
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from image_tagger.class_loader import load_classes
from image_tagger.preprocess import preprocess_image

_detector_model = None
_weights = None

def detect_objects(
    image_path: str,
    classes_path: str = "image_tagger/classes.txt",
    output_path: str = "output_detected.jpg",
    conf_threshold: float = 0.6,
) -> list[str]:
    global _detector_model, _weights

    _ = load_classes(classes_path)
    _ = preprocess_image(image_path, size=(416, 416))

    orig_image = cv2.imread(image_path)
    if orig_image is None:
        raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {image_path}")

    if _detector_model is None:
        _weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        _detector_model = fasterrcnn_resnet50_fpn_v2(weights=_weights)
        _detector_model.eval()

    categories = _weights.meta["categories"]
    rgb_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    tensor_img = torch.from_numpy(rgb_image).permute(2, 0, 1).float() / 255.0

    with torch.no_grad():
        predictions = _detector_model([tensor_img])[0]

    detected_labels = []
    for box, label_idx, score in zip(predictions["boxes"], predictions["labels"], predictions["scores"]):
        conf = float(score)
        if conf >= conf_threshold:
            x1, y1, x2, y2 = map(int, box.tolist())
            label = categories[int(label_idx)]
            detected_labels.append(label)

            cv2.rectangle(orig_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text = f"{label} ({conf:.2f})"
            cv2.putText(orig_image, text, (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imwrite(output_path, orig_image)
    return list(dict.fromkeys(detected_labels))

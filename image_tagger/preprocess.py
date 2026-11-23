import cv2
import numpy as np

def preprocess_image(image_path, size=(224, 224)):
    image = cv2.imread(image_path)

    if image is None:
       raise FileNotFoundError(image_path)

    image = cv2.resize(image, size)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0

    return image
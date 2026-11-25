import cv2
import numpy as np

def preprocess_image(image_path, size=(224, 224)):
    image = cv2.imread(image_path)

    if image is None:
       raise FileNotFoundError(image_path)

    image = cv2.resize(image, size)

    return image
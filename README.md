# Image-caption-tagger

Automatic image captioning, object detection, and keyword tag extractor using Hugging Face (BLIP, CLIP) and Torchvision / OpenCV.



\## Features

1\. \*\*Image Captioning (`captioner.py`)\*\*: Generates natural language descriptions using `Salesforce/blip-image-captioning-base`.

2\. \*\*Object Detection (`detector.py`)\*\*: Detects objects and saves bounding-box visualization (`output\_detected.jpg`) using Faster R-CNN and OpenCV.

3\. \*\*Hashtag Extraction (`tagger.py`)\*\*: Recommends relevant hashtags via zero-shot classification using `openai/clip-vit-base-patch32`.

## Example Results

| Input Image | Object Detection Result |
| :---: | :---: |
| ![Input](image/example_02.jpg) | ![Output](output_detected.jpg) |


\## Installation

```bash

pip install -r requirements.txt


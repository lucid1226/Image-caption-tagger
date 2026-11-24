from pathlib import Path
from transformers import pipeline

#Hugging Face BLIP이미지 캡셔닝 모델을 불러오는 부분
captioner = pipeline(
    "image-to-text",
    model="Salesforce/blip-image-captioning-base"
)

#이미지 파일을 받아서 문장을 생성
def generate_caption(image_path):

  #이미지 파일의 존재를 확인
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError("Image file not found.")

   #BLIP모델을 활용해 이미지 설명을 생성
    result = captioner(str(image_path))

   #설명 문장 가져오기
    if not result or not result[0]["generated_text"]:
        raise ValueError("Caption generation failed.")

    return result[0]["generated_text"].strip()

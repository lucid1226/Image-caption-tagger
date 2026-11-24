from image_tagger.captioner import generate_caption

#테스트할 이미지 경로 
image_path = "image/test.jpg"

#가져온 함수를 실행하여 설명 문장 받기
caption = generate_caption(image_path)

#결과
print("Image Caption:")
print(caption)
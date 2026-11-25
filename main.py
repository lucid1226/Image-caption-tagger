import sys
from pathlib import Path
from image_tagger.captioner import generate_caption
from image_tagger.detector import detect_objects
from image_tagger.tagger import extract_tags

def run_pipeline(image_path: str, output_path: str = "output_detected.jpg"):
    if not Path(image_path).is_file():
        print(f"[오류] 이미지 파일을 찾을 수 없습니다: {image_path}")
        return

    print(f"\n[분석 시작] 대상 이미지: {image_path}")
    print("-" * 50)

    # 1. BLIP 이미지 캡셔닝 (팀원 1 모듈)
    caption = generate_caption(image_path)
    print(f"1. 이미지 설명 (BLIP): {caption}")

    # 2. YOLOv3 객체 탐지 + OpenCV 시각화 (조장 + 팀원 2 모듈)
    objects = detect_objects(image_path, output_path=output_path)
    print(f"2. 탐지된 사물 (YOLOv3): {', '.join(objects) if objects else '탐지된 사물 없음'}")
    print(f"   -> 시각화 결과 저장 완료: {output_path}")

    # 3. CLIP 세부 키워드 해시태그 추천 (조장 모듈)
    hashtags = extract_tags(image_path)
    print(f"3. 추천 해시태그 (CLIP): {' '.join(hashtags)}")
    print("-" * 50)

if __name__ == "__main__":
    # 실행 시 이미지 경로를 인자로 주거나, 기본 image/ 폴더의 첫 번째 이미지 사용
    if len(sys.argv) > 1:
        target_image = sys.argv[1]
    else:
        sample_images = [
            p for p in Path("image").glob("*.*")
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        ]
        if not sample_images:
            print("사용법: python main.py <이미지_경로>")
            sys.exit(1)
        target_image = str(sample_images[0])

    run_pipeline(target_image)
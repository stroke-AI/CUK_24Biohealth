import numpy as np
import cv2
from keras.models import load_model

# 모델 로드
model = load_model('cnn_model.h5')

def preprocess_image_for_cnn(image_path):
    """CNN 모델을 위한 이미지 전처리."""
    if not isinstance(image_path, str):
        raise ValueError(f"Expected a string for image_path, got {type(image_path)}.")
    
    print(f"Loading image for CNN from {image_path}")
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Image at {image_path} could not be loaded.")
    image = cv2.resize(image, (128, 128))
    image = image / 255.0  # 픽셀 값을 [0, 1]로 정규화
    return np.expand_dims(image, axis=0)  # 배치 차원 추가

def display_prediction(prediction_prob):
    """예측 결과와 확신도를 출력합니다."""
    # 확률값 추출
    prob_stroke = prediction_prob[0][0]  # 뇌졸중일 확률
    prob_normal = 1 - prob_stroke  # 정상일 확률
    
    # 확신도 계산
    if prob_stroke > prob_normal:
        confidence = prob_stroke * 100
        result = 'STROKE'
    else:
        confidence = prob_normal * 100
        result = 'NORMAL'
    
    # 결과 출력
    print(f"The image is classified as '{result}' with {confidence:.2f}% confidence.")

    # 경고 메시지
    if confidence < 1.0:
        print("Warning: The confidence is very low, consider validating with additional models or data.")

def main():
    image_path = input("Enter the path to the image: ")
    processed_image = preprocess_image_for_cnn(image_path)
    
    # 예측 수행
    prediction_prob = model.predict(processed_image)
    
    # 결과 출력
    display_prediction(prediction_prob)

if __name__ == "__main__":
    main()

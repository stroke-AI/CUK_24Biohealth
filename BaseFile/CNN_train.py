import numpy as np
import cv2
import os
import math
import dlib
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split

# dlib의 얼굴 랜드마크 예측기 설정
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
SCALE_FACTOR = 1

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

def dist(x, y):
    """두 점 사이의 유클리드 거리 계산."""
    a = x[0,0] - y[0,0]
    b = x[0,1] - y[0,1]
    return math.sqrt((a * a) + (b * b))

def get_landmarks(image):
    """이미지에서 얼굴 랜드마크 추출."""
    rects = detector(image, 1)
    if len(rects) > 1 or len(rects) == 0:
        return None
    return np.matrix([[p.x, p.y] for p in predictor(image, rects[0]).parts()])

def calculate_asymmetry_ratio(image_path):
    """이미지에서 얼굴 랜드마크를 추출하여 비대칭 지표를 계산."""
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        print(f"Image not loaded: {image_path}")
        return None  # 이미지가 로드되지 않은 경우
    
    image = cv2.resize(image, (image.shape[1] * SCALE_FACTOR, image.shape[0] * SCALE_FACTOR))
    landmarks = get_landmarks(image)
    if landmarks is None:
        print(f"Landmarks not found: {image_path}")
        return None  # 랜드마크가 없으면 불완전한 데이터로 간주

    CENTER = landmarks[33]
    LEFT_EYE = landmarks[36]
    RIGHT_EYE = landmarks[45]
    LEFT_NOSE = landmarks[31]
    RIGHT_NOSE = landmarks[35]
    LEFT_LIP = landmarks[48]
    RIGHT_LIP = landmarks[54]

    # 비대칭 지표 계산
    left_eye_dist = dist(CENTER, LEFT_EYE)
    right_eye_dist = dist(CENTER, RIGHT_EYE)
    eye_ratio = min(left_eye_dist, right_eye_dist) / max(left_eye_dist, right_eye_dist)

    left_nose_dist = dist(CENTER, LEFT_NOSE)
    right_nose_dist = dist(CENTER, RIGHT_NOSE)
    nose_ratio = min(left_nose_dist, right_nose_dist) / max(left_nose_dist, right_nose_dist)

    left_lip_dist = dist(CENTER, LEFT_LIP)
    right_lip_dist = dist(CENTER, RIGHT_LIP)
    lip_ratio = min(left_lip_dist, right_lip_dist) / max(left_lip_dist, right_lip_dist)

    total_ratio = eye_ratio + nose_ratio + lip_ratio

    return total_ratio 

def preprocess_image_for_cnn(image_path):
    """CNN 모델을 위한 이미지 전처리."""
    if not isinstance(image_path, str):
        raise ValueError(f"Expected a string for image_path, got {type(image_path)}.")
    
    print(f"Loading image for CNN from {image_path}")
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Image at {image_path} could not be loaded.")
    image = cv2.resize(image, (128, 128))
    return image

def plot_confusion_matrix(y_true, y_pred, title):
    """Confusion matrix를 생성하고 시각화합니다."""
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title)
    plt.show()

def plot_roc_auc_curve(y_true, y_pred_prob, model_name):
    """ROC Curve와 AUC를 시각화합니다."""
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'{model_name} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')
    plt.show()

def build_and_train_cnn(image_paths, y_labels):
    """CNN 모델을 정의하고 학습합니다."""
    
    # 이미지 전처리
    X_images = []
    for img_path in image_paths:
        try:
            processed_image = preprocess_image_for_cnn(img_path)
            X_images.append(processed_image)
        except ValueError as e:
            print(e)  # 처리할 수 없는 이미지에 대한 오류 로그

    if not X_images:
        raise ValueError("No images were processed successfully. Check your image paths and preprocessing function.")

    X_images = np.array(X_images)

    # 픽셀 값을 [0, 1]로 정규화
    X_images = X_images / 255.0

    # 데이터 분할
    X_train_images, X_test_images, y_train, y_test = train_test_split(X_images, y_labels, test_size=0.3, random_state=42)

    # CNN 모델 정의
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

    # 모델 학습
    history = model.fit(
        X_train_images, y_train,
        epochs=10,
        validation_split=0.2,
        batch_size=32,
        verbose=1
    )

    # 모델 평가
    y_pred_prob = model.predict(X_test_images)
    y_pred = (y_pred_prob > 0.5).astype(int)  # 확률을 클래스 레이블로 변환

    # 평가 지표 출력
    print(classification_report(y_test, y_pred, zero_division=0))

    # 학습 이력 시각화
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 2])
    plt.legend(loc='lower right')
    plt.show()

    # Confusion Matrix 시각화
    plot_confusion_matrix(y_test, y_pred, "CNN Model")

    # ROC Curve 및 AUC 시각화
    plot_roc_auc_curve(y_test, y_pred_prob, "CNN Model")

    # 모델 저장
    #model.save('cnn_model.h5')

def load_image_paths_and_labels(odd_path_dir, normal_path_dir):
    """주어진 디렉토리에서 이미지를 로드하고 레이블을 생성합니다."""
    files = []
    t_data = []

    odd_file_list = os.listdir(odd_path_dir)
    normal_file_list = os.listdir(normal_path_dir)
    
    for file_name in odd_file_list:
        image_path = os.path.join(odd_path_dir, file_name)
        asymmetry_ratio = calculate_asymmetry_ratio(image_path)
        if asymmetry_ratio is None:
            print(f"Skipping {image_path}: Unable to process image.")
            continue
        files.append(image_path)
        t_data.append(1)  # 'odd' 이미지 레이블

    for file_name in normal_file_list:
        image_path = os.path.join(normal_path_dir, file_name)
        asymmetry_ratio = calculate_asymmetry_ratio(image_path)
        if asymmetry_ratio is None:
            print(f"Skipping {image_path}: Unable to process image.")
            continue
        files.append(image_path)
        t_data.append(0)  # 'normal' 이미지 레이블

    X = np.array(files)  # 이미지 경로를 문자열로 저장
    y = np.array(t_data)  # 레이블을 1D 배열로 저장
    return X, y

# 데이터 로드 및 전처리
X_paths, y_labels = load_image_paths_and_labels('data/odd_pictures', 'data/normal_pictures')

# CNN 학습
print("Training CNN model...")
build_and_train_cnn(X_paths, y_labels)

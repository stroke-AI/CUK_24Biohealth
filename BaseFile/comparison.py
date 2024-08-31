import numpy as np
import cv2
import os
import math
import dlib
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
import seaborn as sns
import matplotlib.pyplot as plt

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

def load_image_paths_and_labels(odd_path_dir, normal_path_dir):
    """이미지 파일을 로드하고 레이블을 생성합니다."""
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

def preprocess_asymmetry_features(image_paths):
    """이미지 경로에서 비대칭 지표를 추출하고 특성 배열을 생성합니다."""
    features = []
    for img_path in image_paths:
        try:
            ratio = calculate_asymmetry_ratio(img_path)
            if ratio is not None:
                features.append([ratio])
        except ValueError as e:
            print(e)  # 처리할 수 없는 이미지에 대한 오류 로그
    return np.array(features)

def flatten_images(X_images):
    """3D 이미지 배열을 2D 배열로 평탄화."""
    return X_images.reshape((X_images.shape[0], -1))

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

def train_and_evaluate_model_with_cm(model_class, X_train, y_train, X_test, y_test, title, **kwargs):
    """다양한 머신러닝 모델을 학습하고 평가합니다. Confusion Matrix와 ROC Curve를 시각화합니다."""
    model = model_class(**kwargs)

    # 필요한 경우 이미지 평탄화
    if model_class in [LogisticRegression, MLPClassifier, SVC, DecisionTreeClassifier, RandomForestClassifier, GradientBoostingClassifier]:
        X_train = flatten_images(X_train)
        X_test = flatten_images(X_test)

    # 클래스 가중치 조정
    if model_class in [LogisticRegression, MLPClassifier, SVC, DecisionTreeClassifier, RandomForestClassifier, GradientBoostingClassifier]:
        class_weights = {0: 1, 1: 2}  # 불균형 데이터셋을 위한 가중치 조정
        kwargs['class_weight'] = class_weights

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_pred_prob = model.predict_proba(X_test)[:, 1]
        plot_roc_auc_curve(y_test, y_pred_prob, title)
    else:
        y_pred_prob = np.zeros_like(y_pred)     

    # 평가 지표 계산
    accuracy = np.mean(y_pred == y_test)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    # 결과 출력
    print(f"정확도: {accuracy * 100:.2f}%")
    print(f"정밀도: {precision * 100:.2f}%")
    print(f"재현율: {recall * 100:.2f}%")
    print(f"F1 점수: {f1 * 100:.2f}%")

    # 분류 보고서 출력
    print("\n분류 보고서:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Confusion Matrix 시각화
    plot_confusion_matrix(y_test, y_pred, title)

# 데이터 로드 및 전처리
X_paths, y_labels = load_image_paths_and_labels('data/odd_pictures', 'data/normal_pictures')

# 비대칭 특성 추출
X_features = preprocess_asymmetry_features(X_paths)

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X_features, y_labels, test_size=0.3, random_state=42)

# 다양한 모델 학습 및 평가
print("Logistic Regression:")
train_and_evaluate_model_with_cm(LogisticRegression, X_train, y_train, X_test, y_test, "Logistic Regression", max_iter=10000)

print("\nSupport Vector Machine (SVM):")
train_and_evaluate_model_with_cm(SVC, X_train, y_train, X_test, y_test, "Support Vector Machine (SVM)", kernel='rbf', C=1.0, gamma='scale')

print("\nDecision Tree:")
train_and_evaluate_model_with_cm(DecisionTreeClassifier, X_train, y_train, X_test, y_test, "Decision Tree")

print("\nRandom Forest:")
train_and_evaluate_model_with_cm(RandomForestClassifier, X_train, y_train, X_test, y_test, "Random Forest", n_estimators=100)

print("\nGradient Boosting:")
train_and_evaluate_model_with_cm(GradientBoostingClassifier, X_train, y_train, X_test, y_test, "Gradient Boosting")

print("\nNeural Network:")
train_and_evaluate_model_with_cm(MLPClassifier, X_train, y_train, X_test, y_test, "Neural Network", hidden_layer_sizes=(100,), max_iter=500)
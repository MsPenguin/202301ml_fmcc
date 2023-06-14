import numpy as np
from sklearn.utils import shuffle
from utils import load_dataset
from utils import mfcc_feature
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle

# 상수값들을 정의합니다.
SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2

TRAIN_DATASET_BASE = "raw16k/train/"

# 훈련데이터를 불러옵니다.
train_audio_male, train_audio_female = load_dataset.load_trainset(
  TRAIN_DATASET_BASE, SAMPLE_RATE, SAMPLE_WIDTH)

# 훈련데이터에서 MFCC 특징을 추출합니다.
train_feature_male = mfcc_feature.extract(train_audio_male, SAMPLE_RATE)
train_feature_female = mfcc_feature.extract(train_audio_female, SAMPLE_RATE)

# 훈련데이터 차원과 레이블을 확인합니다.
assert train_feature_male.shape[0] == train_feature_female.shape[0], "남녀 훈련 데이터 비율이 같아야 합니다."

# 훈련데이터의 레이블을 생성합니다.
train_label_male = np.ones(train_feature_male.shape[0])
train_label_female = np.zeros(train_feature_female.shape[0])

# 훈련데이터를 합치고 섞습니다.
train_all_features = np.vstack((train_feature_male, train_feature_female))
train_all_labels = np.concatenate((train_label_male, train_label_female))

train_all_features, train_all_labels = shuffle(train_all_features, train_all_labels)

# 클래스가 올바르게 생성되었는지 확인합니다.
assert len(np.unique(train_all_labels)) == 2, "클래스는 2개여야 합니다."

# 클래스 불균형 여부를 확인합니다.
unique_labels, label_counts = np.unique(train_all_labels, return_counts=True)
assert all(count > 0 for count in label_counts), "각 클래스는 적어도 하나 이상의 샘플이 있어야 합니다."
assert np.max(label_counts) / np.min(label_counts) <= 2, "클래스 불균형이 발생하였습니다. 데이터를 다시 확인해주세요."

# SVM 모델을 정의합니다.
classifier = SVC()

# 학습 데이터의 크기를 10%부터 100%까지 10%씩 증가시키면서 학습합니다.
training_sizes = np.linspace(0.1, 1.0, 10)

# Accuracy를 저장할 리스트를 생성합니다.
classifiers = []
train_accuracies = []

# 학습을 시작합니다.
for size in training_sizes:
  # 샘플의 개수를 계산합니다.
  num_samples = int(size * train_all_features.shape[0])
  
  # 훈련 데이터의 부분집합을 구합니다.
  X_subset = train_all_features[:num_samples]
  y_subset = train_all_labels[:num_samples]
  
  # 분류기를 학습시킵니다.
  classifier.fit(X_subset, y_subset)
  
  # 훈련 데이터에서의 정확도를 계산합니다.
  y_train_pred = classifier.predict(X_subset)
  train_accuracy = accuracy_score(y_subset, y_train_pred)
  
  # 분류기와 정확도를 리스트에 저장합니다.
  classifiers.append(classifier)
  train_accuracies.append(train_accuracy)

# 학습된 모델을 저장합니다.
# with open("trained_model.pkl", "wb") as f:
#   pickle.dump((classifiers, train_accuracies), f)

# 특정 분류기만 내보냅니다.
with open("final_model.pkl", "wb") as f:
  pickle.dump((classifiers[6]), f)
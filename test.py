import numpy as np
from utils import load_dataset
from utils import mfcc_feature
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pickle

# 상수값들을 정의합니다.
SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2

TEST_DATASET_BASE = "raw16k/test/"

# 테스트데이터를 불러옵니다.
test_audio = load_dataset.load_testset(TEST_DATASET_BASE, SAMPLE_RATE, SAMPLE_WIDTH)

# 테스트데이터에서 MFCC 특징을 추출합니다.
test_feature = mfcc_feature.extract(test_audio, SAMPLE_RATE)

# 테스트 데이터의 참 레이블을 생성합니다.
y_true = np.concatenate((np.zeros((test_feature.shape[0] // 2 - 50,), dtype=int), np.ones((test_feature.shape[0] // 2 + 50,), dtype=int)))

# 학습된 모델들을 불러옵니다.
# with open("trained_model.pkl", "rb") as f:
#   classifiers, train_accuracies = pickle.load(f)

# 최종 모델을 불러옵니다.
with open("final_model.pkl", "rb") as f:
  classifier = pickle.load(f)

# testing_size = len(classifiers)

# 테스트 데이터에서의 정확도를 계산합니다.
# classifier = classifiers[6]  # 원하는 모델을 선택합니다.
y_val_pred = classifier.predict(test_feature)
val_accuracy = accuracy_score(y_true, y_val_pred)
print(val_accuracy)

TAG = 'fmcc_test_'
result = []
for i in range(len(y_val_pred)):
  if y_val_pred[i] == 0:
    result.append(TAG + str(i + 1).zfill(4) +' feml')
  else:
    result.append(TAG + str(i + 1).zfill(4) +' male')

with open('result.txt', 'w') as f:
  for item in result:
    f.write("%s\n" % item)

# training_sizes = np.linspace(0.1, 1.0, 10)
# plt.plot(training_sizes[:testing_size], train_accuracies[:testing_size], label='Training Accuracy')
# plt.plot(training_sizes[:testing_size], test_accuracies[:testing_size], label='Test Accuracy')
# plt.xlabel('Training Set Size')
# plt.ylabel('Accuracy')
# plt.title('Learning Curve')
# plt.legend()
# plt.show()

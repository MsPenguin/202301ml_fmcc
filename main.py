import numpy as np
from utils import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from python_speech_features import mfcc

TRAIN_PATH_LIST = "fmcc_train.ctl"
TEST_PATH_LIST = "fmcc_test.ctl"

TRAIN_DATASET_BASE = "raw16k/train/"
TEST_DATASET_BASE = "raw16k/test/"

# Step 1: Collect and prepare the dataset
# Load your voice dataset and split it into features (X) and labels (y)

audio_male = []  # 남성 음성 데이터
audio_female = []  # 여성 음성 데이터

audio_male, audio_female = load_dataset.load(TRAIN_DATASET_BASE)

print(audio_female)


# # Step 2: Feature extraction
# # Extract MFCC features from the voice samples
# def extract_features(audio, sample_rate):
#     mfcc_features = mfcc(audio, sample_rate)
#     return mfcc_features

# # Step 3: Training the classifier
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Extract MFCC features for training data
# X_train_features = np.array([extract_features(audio, sample_rate) for audio, sample_rate in X_train])

# # Initialize and train the classifier
# classifier = SVC()
# classifier.fit(X_train_features, y_train)

# # Step 4: Model evaluation
# # Extract MFCC features for testing data
# X_test_features = np.array([extract_features(audio, sample_rate) for audio, sample_rate in X_test])

# # Make predictions on the testing data
# y_pred = classifier.predict(X_test_features)

# # Evaluate the model
# accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy:", accuracy)

# # Step 5: Prediction
# # Extract MFCC features for a new voice sample
# new_sample_features = np.array([extract_features(new_sample_audio, new_sample_sample_rate)])

# # Make prediction on the new sample
# predicted_class = classifier.predict(new_sample_features)
# print("Predicted class:", predicted_class)
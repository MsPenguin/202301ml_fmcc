import numpy as np
from utils import load_dataset
import librosa
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from python_speech_features import mfcc

SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2

TRAIN_DATASET_BASE = "raw16k/train/"
TEST_DATASET_BASE = "raw16k/test/"

# Step 1: Collect and prepare the dataset
# Load your voice dataset and split it into features (X) and labels (y)

audio_male = []  # 남성 음성 데이터
audio_female = []  # 여성 음성 데이터

audio_male, audio_female = load_dataset.load(TRAIN_DATASET_BASE, SAMPLE_RATE, SAMPLE_WIDTH)

print(audio_female)


# # Step 2: Feature extraction
# # Extract MFCC features from the voice samples
def extract_feature(audio, sample_rate):
    mfcc_feature = librosa.feature.mfcc(audio, sample_rate)
    return mfcc_feature

feature_male = np.array([extract_feature(audio) for audio in audio_male])
feature_female = np.array([extract_feature(audio) for audio in audio_female])

label_male = np.ones(len(feature_male))
label_female = np.zeros(len(feature_female))

# Concatenate positive and negative features and labels
all_features = np.concatenate((feature_male, feature_female))
all_labels = np.concatenate((label_male, label_female))

# # Step 3: Training the classifier
X_train, X_test, y_train, y_test = train_test_split(all_features, all_labels, test_size=0.2, random_state=42)

# # Initialize and train the classifier
classifier = SVC()
classifier.fit(all_features, y_train)

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
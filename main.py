import numpy as np
from utils import load_dataset
from utils import mfcc_feature
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from python_speech_features import mfcc

SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2

TRAIN_DATASET_BASE = "raw16k/train/"
TEST_DATASET_BASE = "raw16k/test/"

# Step 1: Collect and prepare the dataset

train_audio_male, train_audio_female = load_dataset.load_trainset(TRAIN_DATASET_BASE, SAMPLE_RATE, SAMPLE_WIDTH)
test_audio = load_dataset.load_testset(TEST_DATASET_BASE, SAMPLE_RATE, SAMPLE_WIDTH)



# # Step 2: Feature extraction
# # Extract MFCC features from the voice samples

train_feature_male = mfcc_feature.extract(train_audio_male, SAMPLE_RATE)
train_feature_female = mfcc_feature.extract(train_audio_female, SAMPLE_RATE)
test_feature = mfcc_feature.extract(test_audio, SAMPLE_RATE)


train_label_male = [1] * len(train_feature_male)
train_label_female = [0] * len(train_feature_female)

# Concatenate positive and negative features and labels
train_all_features = train_feature_male + train_feature_female
train_all_labels = train_label_male + train_label_female


# Initialize and train the classifier
classifier = SVC()
classifier.fit(train_all_features, train_all_labels)

# Make predictions on the testing data
y_pred = classifier.predict(test_audio)
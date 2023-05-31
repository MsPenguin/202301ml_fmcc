import numpy as np
from utils import load_dataset
from utils import mfcc_feature
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2

TRAIN_DATASET_BASE = "raw16k/train/"
TEST_DATASET_BASE = "raw16k/test/"

# Step 1: Collect and prepare the dataset

train_audio_male, train_audio_female = load_dataset.load_trainset(
    TRAIN_DATASET_BASE, SAMPLE_RATE, SAMPLE_WIDTH)

test_audio = load_dataset.load_testset(TEST_DATASET_BASE, SAMPLE_RATE, SAMPLE_WIDTH)


# Step 2: Feature extraction
# Extract MFCC features from the voice samples

train_feature_male = mfcc_feature.extract(train_audio_male, SAMPLE_RATE)
train_feature_female = mfcc_feature.extract(train_audio_female, SAMPLE_RATE)

test_feature = mfcc_feature.extract(test_audio, SAMPLE_RATE)

# Check dimensions and labels of the training data
assert train_feature_male.shape[0] == train_feature_female.shape[0], "Number of male and female features should be the same"

train_label_male = np.ones(train_feature_male.shape[0])
train_label_female = np.zeros(train_feature_female.shape[0])

# Concatenate positive and negative features and labels
train_all_features = np.vstack((train_feature_male, train_feature_female))
train_all_labels = np.concatenate((train_label_male, train_label_female))

# Check if the number of classes is at least 2
assert len(np.unique(train_all_labels)) >= 2, "Number of classes should be greater than one"

# Check if the number of samples for each class is balanced
unique_labels, label_counts = np.unique(train_all_labels, return_counts=True)
assert all(count > 0 for count in label_counts), "Each class should have at least one sample"
assert np.max(label_counts) / np.min(label_counts) <= 2, "Class imbalance detected, please balance the training data"

# Initialize and train the classifier
classifier = SVC()

# Define the range of training set sizes as percentages
training_sizes = np.linspace(0.1, 1.0, 10)

# Initialize lists to store accuracies
train_accuracies = []
val_accuracies = []

# True labels for the test data
y_true = np.concatenate((np.zeros((test_feature.shape[0] // 2,), dtype=int), np.ones((test_feature.shape[0] // 2,), dtype=int)))

# Iterate over different training set sizes
for size in training_sizes:
    # Determine the number of samples based on the training size
    num_samples = int(size * train_all_features.shape[0])
    
    # Select a subset of the training data and labels
    X_subset = train_all_features[:num_samples]
    y_subset = train_all_labels[:num_samples]
    
    # Train the SVM model
    classifier.fit(X_subset, y_subset)
    
    # Make predictions on the training data
    y_train_pred = classifier.predict(X_subset)
    train_accuracy = accuracy_score(y_subset, y_train_pred)
    
    # Make predictions on the validation data
    y_val_pred = classifier.predict(test_feature)
    val_accuracy = accuracy_score(y_true, y_val_pred)
    
    # Append the accuracies to the lists
    train_accuracies.append(train_accuracy)
    val_accuracies.append(val_accuracy)

# Plot the learning curve
plt.plot(training_sizes, train_accuracies, label='Training Accuracy')
plt.plot(training_sizes, val_accuracies, label='Validation Accuracy')
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy')
plt.title('Learning Curve')
plt.legend()
plt.show()
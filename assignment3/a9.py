import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

dataframe = pd.read_csv("DCT_mal.csv", header=None, low_memory=False)

feature_vectors = dataframe.iloc[:, :-1].values
class_labels = dataframe.iloc[:, -1].values

chosen_classes = ['3368', '3364']
selection_mask = np.isin(class_labels, chosen_classes)
filtered_features = feature_vectors[selection_mask]
filtered_labels = class_labels[selection_mask]

training_features, testing_features, training_labels, testing_labels = train_test_split(
    filtered_features, filtered_labels, test_size=0.3, random_state=42
)

knn_classifier = KNeighborsClassifier(n_neighbors=3)
knn_classifier.fit(training_features, training_labels)

training_predictions = knn_classifier.predict(training_features)
testing_predictions = knn_classifier.predict(testing_features)

training_accuracy_score = accuracy_score(training_labels, training_predictions)
testing_accuracy_score = accuracy_score(testing_labels, testing_predictions)

print(f"Training Accuracy: {training_accuracy_score*100:.2f}%")
print(f"Test Accuracy: {testing_accuracy_score*100:.2f}%")

training_confusion_matrix = confusion_matrix(training_labels, training_predictions, labels=chosen_classes)
print("\nConfusion Matrix - Training Data:")
print(training_confusion_matrix)

testing_confusion_matrix = confusion_matrix(testing_labels, testing_predictions, labels=chosen_classes)
print("\nConfusion Matrix - Test Data:")
print(testing_confusion_matrix)

print("\nClassification Report - Training Data:")
print(classification_report(training_labels, training_predictions, target_names=chosen_classes))

print("\nClassification Report - Test Data:")
print(classification_report(testing_labels, testing_predictions, target_names=chosen_classes))

plt.figure(figsize=(6,5))
sns.heatmap(testing_confusion_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=chosen_classes, yticklabels=chosen_classes)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix - Test Data')
plt.show()
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

dataframe = pd.read_csv("DCT_mal.csv", header=None)

feature_vectors = dataframe.iloc[:, :-1].values
class_labels = dataframe.iloc[:, -1].values

chosen_classes = [3368, 3364]
selection_mask = np.isin(class_labels, chosen_classes)

filtered_features = feature_vectors[selection_mask]
filtered_labels = class_labels[selection_mask].astype(str)

training_features, testing_features, training_labels, testing_labels = train_test_split(
    filtered_features, filtered_labels, test_size=0.3, random_state=42
)

k_values = range(1, 12)
accuracy_scores = []

for neighbor_count in k_values:
    knn_model = KNeighborsClassifier(n_neighbors=neighbor_count)
    knn_model.fit(training_features, training_labels)
    score = knn_model.score(testing_features, testing_labels)
    accuracy_scores.append(score)

plt.figure(figsize=(8,6))
plt.plot(k_values, accuracy_scores, marker='o', color='blue')
plt.title('kNN Accuracy vs k')
plt.xlabel('k (Number of Neighbors)')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()
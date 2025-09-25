import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

dataframe = pd.read_csv("DCT_mal.csv")

feature_vectors = dataframe.iloc[:, :-1].values
class_labels = dataframe.iloc[:, -1].values

chosen_classes = [3368, 3364]
selection_mask = np.isin(class_labels, chosen_classes)

filtered_features = feature_vectors[selection_mask]
filtered_labels = class_labels[selection_mask].astype(str)

training_features, testing_features, training_labels, testing_labels = train_test_split(
    filtered_features, filtered_labels, test_size=0.3, random_state=42
)

knn_classifier = KNeighborsClassifier(n_neighbors=3)
knn_classifier.fit(training_features, training_labels)

print("Model trained successfully with k=3.")
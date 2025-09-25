import numpy as np
import pandas as pd

dataset_path = "DCT_mal.csv"
dataframe = pd.read_csv(dataset_path)

feature_vectors = dataframe.iloc[:, :-1].values
class_labels = dataframe.iloc[:, -1].values

distinct_classes = np.unique(class_labels)
first_class = distinct_classes[0]
second_class = distinct_classes[1]

data_for_first_class = feature_vectors[class_labels == first_class]
data_for_second_class = feature_vectors[class_labels == second_class]

mean_vector_1 = np.mean(data_for_first_class, axis=0)
mean_vector_2 = np.mean(data_for_second_class, axis=0)

std_dev_1 = np.std(data_for_first_class, axis=0)
std_dev_2 = np.std(data_for_second_class, axis=0)

distance_between_centroids = np.linalg.norm(mean_vector_1 - mean_vector_2)

print(f"Centroid of Class {first_class}:\n{mean_vector_1}\n")
print(f"Centroid of Class {second_class}:\n{mean_vector_2}\n")
print(f"Spread (Standard Deviation) of Class {first_class}:\n{std_dev_1}\n")
print(f"Spread (Standard Deviation) of Class {second_class}:\n{std_dev_2}\n")
print(f"Inter-class Distance between Class {first_class} and Class {second_class}: {distance_between_centroids}\n")
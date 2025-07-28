import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset_path = "DCT_mal.csv"
dataframe = pd.read_csv(dataset_path, header=None)

selected_feature_column = 0
data_from_feature = dataframe.iloc[:, selected_feature_column].values

calculated_mean = np.mean(data_from_feature)
calculated_variance = np.var(data_from_feature)

print(f"Feature Selected: Column Index {selected_feature_column}")
print(f"Mean: {calculated_mean}")
print(f"Variance: {calculated_variance}")

histogram_counts, histogram_bin_edges = np.histogram(data_from_feature, bins=10)
print(f"Histogram Bins Counts: {histogram_counts}")
print(f"Histogram Bin Edges: {histogram_bin_edges}")

plt.figure(figsize=(8, 6))
plt.hist(data_from_feature, bins=8, color='skyblue', edgecolor='black')
plt.title(f'Histogram of Feature: Column {selected_feature_column}')
plt.xlabel('Feature Value Ranges')
plt.ylabel('Frequency Count')
plt.grid(True)
plt.show()
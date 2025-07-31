import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset_path = "DCT_mal.csv"
dataframe = pd.read_csv(dataset_path, header=None)

data_vector_a = dataframe.iloc[0, :-1].values
data_vector_b = dataframe.iloc[1, :-1].values

minkowski_orders = np.arange(1, 11)
calculated_distances = []

for order in minkowski_orders:
    distance = np.sum(np.abs(data_vector_a - data_vector_b) ** order) ** (1 / order)
    calculated_distances.append(distance)

plt.figure(figsize=(8, 6))
plt.plot(minkowski_orders, calculated_distances, marker='o', color='purple')
plt.title('Minkowski Distance vs r (between two vectors)')
plt.xlabel('r (Order of Minkowski Distance)') 
plt.ylabel('Distance')
plt.grid(True)
plt.show()
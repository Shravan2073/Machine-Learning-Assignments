# a5_clustering.py

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score

# Load the dataset
try:
    data = pd.read_csv('DCT_mal.csv')
except FileNotFoundError:
    print("Error: DCT_mal.csv not found. Please ensure the file is in the same directory.")
    exit()

# For clustering, we use only the features (independent variables)
X = data.drop('LABEL', axis=1)

# Scale the data for better clustering performance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reduce dimensionality to 2 components for visualization
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# --- 1. Hierarchical Clustering ---
print("--- Applying Hierarchical Clustering ---")
# Assuming 2 clusters since the original problem was binary
agg_clustering = AgglomerativeClustering(n_clusters=2)
labels_agg = agg_clustering.fit_predict(X_scaled)
silhouette_agg = silhouette_score(X_scaled, labels_agg)
print(f"Silhouette Score for Hierarchical Clustering: {silhouette_agg:.4f}")

# --- 2. Density-Based Clustering (DBSCAN) ---
print("\n--- Applying Density-Based Clustering (DBSCAN) ---")
# These parameters may need tuning for optimal results
dbscan = DBSCAN(eps=5.0, min_samples=5)
labels_dbscan = dbscan.fit_predict(X_scaled)
n_clusters = len(set(labels_dbscan)) - (1 if -1 in labels_dbscan else 0)
n_noise = list(labels_dbscan).count(-1)
print(f'DBSCAN Estimated number of clusters: {n_clusters}')
print(f'DBSCAN Estimated number of noise points: {n_noise}')

if n_clusters > 1:
    silhouette_dbscan = silhouette_score(X_scaled, labels_dbscan)
    print(f"Silhouette Score for DBSCAN: {silhouette_dbscan:.4f}")
else:
    print("Cannot compute Silhouette Score for DBSCAN with one or zero clusters.")

# --- Visualization ---
print("\nGenerating cluster visualizations... 🖼️")
plt.figure(figsize=(14, 7))

# Plot for Hierarchical Clustering
plt.subplot(1, 2, 1)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels_agg, cmap='viridis', s=10)
plt.title('Hierarchical Clustering Results (2 Clusters)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True)

# Plot for DBSCAN
plt.subplot(1, 2, 2)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels_dbscan, cmap='plasma', s=10)
plt.title('DBSCAN Clustering Results')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True)

plt.tight_layout()
plt.show()
print("Please close the plot window to exit.")
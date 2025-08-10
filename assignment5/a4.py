"""
A4: Perform KMeans clustering (k=2 by default) on features (target column removed).
Print cluster centers and labels summary.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

CSV_PATH = "DCT_mal.csv"
TARGET_COL = None
K = 2
RANDOM_STATE = 42

def load_features_for_clustering(path, target_col=None):
    df = pd.read_csv(path).dropna().reset_index(drop=True)
    if target_col is None:
        for name in ["target", "y", "label", "class"]:
            if name in df.columns:
                target_col = name
                break
    if target_col is None:
        target_col = df.columns[-1]
    # Use numeric features only
    numeric_df = df.select_dtypes(include=[np.number]).copy()
    if target_col in numeric_df.columns:
        numeric_df = numeric_df.drop(columns=[target_col])
    if numeric_df.shape[1] == 0:
        raise ValueError("No numeric features available for clustering.")
    return numeric_df.values, numeric_df.columns.tolist()

def kmeans_fit(X, n_clusters=2, random_state=42):
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto").fit(X)
    return kmeans

def main():
    X, feature_names = load_features_for_clustering(CSV_PATH, TARGET_COL)
    kmeans = kmeans_fit(X, n_clusters=K, random_state=RANDOM_STATE)
    print(f"KMeans with k={K}")
    print("Cluster centers (one row per cluster):")
    for i, center in enumerate(kmeans.cluster_centers_):
        print(f"Cluster {i}: {center}")
    # label counts
    labels, counts = np.unique(kmeans.labels_, return_counts=True)
    print("Cluster label counts:")
    for l, c in zip(labels, counts):
        print(f"  Label {l}: {c}")

if __name__ == "__main__":
    main()

"""
A5: For a given k (default 2), compute Silhouette Score, Calinski-Harabasz (CH) and Davies-Bouldin (DB).
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

CSV_PATH = "DCT_mal.csv"
TARGET_COL = None
K = 2
RANDOM_STATE = 42

def load_features(path, target_col=None):
    df = pd.read_csv(path).dropna().reset_index(drop=True)
    if target_col is None:
        for name in ["target", "y", "label", "class"]:
            if name in df.columns:
                target_col = name
                break
    if target_col is None:
        target_col = df.columns[-1]
    numeric_df = df.select_dtypes(include=[np.number]).copy()
    if target_col in numeric_df.columns:
        numeric_df = numeric_df.drop(columns=[target_col])
    if numeric_df.shape[1] == 0:
        raise ValueError("No numeric features available for clustering.")
    return numeric_df.values

def compute_scores(X, labels):
    sil = silhouette_score(X, labels) if len(np.unique(labels)) > 1 else float("nan")
    ch = calinski_harabasz_score(X, labels) if len(np.unique(labels)) > 1 else float("nan")
    db = davies_bouldin_score(X, labels) if len(np.unique(labels)) > 1 else float("nan")
    return {"Silhouette": sil, "Calinski-Harabasz": ch, "Davies-Bouldin": db}

def main():
    X = load_features(CSV_PATH, TARGET_COL)
    kmeans = KMeans(n_clusters=K, random_state=RANDOM_STATE, n_init="auto").fit(X)
    scores = compute_scores(X, kmeans.labels_)
    print(f"Clustering scores for k={K}:")
    for k, v in scores.items():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    main()

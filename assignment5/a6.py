"""
A6: Run KMeans for a range of k values, compute silhouette, CH and DB for each k,
and print the results. (No plotting in this file; A7 will plot the elbow/inertia.)
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

CSV_PATH = "DCT_mal.csv"
TARGET_COL = None
K_RANGE = range(2, 11)  # change upper bound as needed
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

def compute_scores_for_range(X, k_values):
    results = []
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init="auto").fit(X)
        labels = kmeans.labels_
        sil = silhouette_score(X, labels) if k > 1 else float("nan")
        ch = calinski_harabasz_score(X, labels) if k > 1 else float("nan")
        db = davies_bouldin_score(X, labels) if k > 1 else float("nan")
        results.append({"k": k, "inertia": kmeans.inertia_, "silhouette": sil, "CH": ch, "DB": db})
    return results

def main():
    X = load_features(CSV_PATH, TARGET_COL)
    results = compute_scores_for_range(X, K_RANGE)
    print("k, inertia, silhouette, CH, DB")
    for r in results:
        print(f"{r['k']}, {r['inertia']}, {r['silhouette']}, {r['CH']}, {r['DB']}")

if __name__ == "__main__":
    main()

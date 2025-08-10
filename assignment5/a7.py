"""
A7: Compute distortions (inertia) for k in range and plot elbow curve.
Saves plot as 'elbow_plot.png' in current working directory.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

CSV_PATH = "DCT_mal.csv"
TARGET_COL = None
K_RANGE = range(2, 20)
RANDOM_STATE = 42
OUTPUT_PNG = "elbow_plot.png"

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

def compute_inertia(X, k_values):
    distortions = []
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init="auto").fit(X)
        distortions.append(kmeans.inertia_)
    return distortions

def main():
    X = load_features(CSV_PATH, TARGET_COL)
    distortions = compute_inertia(X, K_RANGE)
    plt.figure(figsize=(8, 5))
    plt.plot(list(K_RANGE), distortions, marker='o')
    plt.xlabel("k (number of clusters)")
    plt.ylabel("Inertia (distortion)")
    plt.title("Elbow Plot for KMeans")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG)
    print(f"Elbow plot saved to {OUTPUT_PNG}")

if __name__ == "__main__":
    main()

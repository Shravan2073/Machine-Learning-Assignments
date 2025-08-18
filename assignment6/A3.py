
import argparse
import numpy as np
import pandas as pd
from collections import Counter

def equal_width_binning(series: pd.Series, n_bins: int = 4) -> pd.Series:
    if not np.issubdtype(series.dropna().dtype, np.number):
        raise ValueError("equal_width_binning expects a numeric series.")
    return pd.cut(series, bins=n_bins, include_lowest=True).astype(str)

def equal_freq_binning(series: pd.Series, n_bins: int = 4) -> pd.Series:
    if not np.issubdtype(series.dropna().dtype, np.number):
        raise ValueError("equal_freq_binning expects a numeric series.")
    unique_vals = series.dropna().unique()
    q = min(n_bins, len(unique_vals))
    return pd.qcut(series, q=q, duplicates="drop").astype(str)

def entropy_from_counts(counts):
    total = sum(counts)
    if total == 0:
        return 0.0
    ent = 0.0
    for c in counts:
        if c == 0: continue
        p = c / total
        ent -= p * np.log2(p)
    return ent

def information_gain(y: pd.Series, x: pd.Series):
    # assumes x is categorical (str) and y is categorical (str)
    y = y.astype(str)
    x = x.astype(str)
    H_y = entropy_from_counts(Counter(y).values())
    total = len(y)
    cond = 0.0
    for v, idx in x.groupby(x).groups.items():
        y_subset = y.loc[idx]
        cond += (len(y_subset) / total) * entropy_from_counts(Counter(y_subset).values())
    return H_y - cond

def main():
    parser = argparse.ArgumentParser(description="A3: Find root node feature by Information Gain.")
    parser.add_argument("--csv", default="DCT_mal.csv", help="Path to CSV.")
    parser.add_argument("--target", default=None, help="Target column (defaults to last).")
    parser.add_argument("--bins", type=int, default=4, help="Number of bins for continuous features.")
    parser.add_argument("--binning", default="equal_width", choices=["equal_width", "equal_freq"], help="Binning method for numeric features.")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    target = args.target or df.columns[-1]
    y = df[target]
    # Convert target to categorical if numeric (bin it for IG computation clarity)
    if np.issubdtype(y.dropna().dtype, np.number):
        y = equal_width_binning(y, n_bins=args.bins)

    features = [c for c in df.columns if c != target]
    ig_scores = {}

    for feat in features:
        col = df[feat]
        if np.issubdtype(col.dropna().dtype, np.number):
            if args.binning == "equal_width":
                x_cat = equal_width_binning(col, n_bins=args.bins)
            else:
                x_cat = equal_freq_binning(col, n_bins=args.bins)
        else:
            x_cat = col.astype(str)
        ig = information_gain(y, x_cat)
        ig_scores[feat] = ig

    best = max(ig_scores, key=ig_scores.get)
    print("Information Gain per feature:")
    for k, v in sorted(ig_scores.items(), key=lambda kv: kv[1], reverse=True):
        print(f"  {k}: {v:.6f}")
    print(f"\nRoot node by Information Gain: {best} (IG={ig_scores[best]:.6f})")

if __name__ == "__main__":
    main()

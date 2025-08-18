
import argparse
import numpy as np
import pandas as pd
from collections import Counter

# Reuse local binning helpers (duplicated for standalone use)
def equal_width_binning(series: pd.Series, n_bins: int = 4) -> pd.Series:
    if not np.issubdtype(series.dropna().dtype, np.number):
        raise ValueError("equal_width_binning expects a numeric series.")
    return pd.cut(series, bins=n_bins, include_lowest=True).astype(str)

def entropy_from_counts(counts):
    total = sum(counts)
    if total == 0:
        return 0.0
    ent = 0.0
    for c in counts:
        if c == 0:
            continue
        p = c / total
        ent -= p * np.log2(p)
    return ent

def entropy(series: pd.Series, treat_continuous_with_bins: bool = True, n_bins: int = 4):
    s = series.dropna()
    if s.empty:
        return 0.0

    if np.issubdtype(s.dtype, np.number) and treat_continuous_with_bins:
        s = equal_width_binning(s, n_bins=n_bins)

    counts = Counter(s)
    return entropy_from_counts(counts.values())

def main():
    parser = argparse.ArgumentParser(description="A1: Calculate entropy of the target column (bin continuous targets if needed).")
    parser.add_argument("--csv", default="DCT_mal.csv", help="Path to CSV.")
    parser.add_argument("--target", default=None, help="Target column name (defaults to last column).")
    parser.add_argument("--bins", type=int, default=4, help="Number of bins for continuous target.")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    target = args.target or df.columns[-1]
    H = entropy(df[target], treat_continuous_with_bins=True, n_bins=args.bins)
    print(f"Entropy of target '{target}' = {H:.6f} bits")

if __name__ == "__main__":
    main()


import argparse
import numpy as np
import pandas as pd

def equal_width_binning(series: pd.Series, n_bins: int = 4, labels=None) -> pd.Series:
    """Bin a numeric pandas Series into equal-width bins using pandas.cut."""
    if not np.issubdtype(series.dropna().dtype, np.number):
        raise ValueError("equal_width_binning expects a numeric series.")
    return pd.cut(series, bins=n_bins, labels=labels, include_lowest=True)

def equal_freq_binning(series: pd.Series, n_bins: int = 4, labels=None) -> pd.Series:
    """Bin a numeric pandas Series into equal-frequency (quantile) bins using pandas.qcut."""
    if not np.issubdtype(series.dropna().dtype, np.number):
        raise ValueError("equal_freq_binning expects a numeric series.")
    # Handle edge cases where fewer unique values than bins
    unique_vals = series.dropna().unique()
    q = min(n_bins, len(unique_vals))
    return pd.qcut(series, q=q, labels=labels, duplicates="drop")

def bin_series(series: pd.Series, method: str = "equal_width", n_bins: int = 4, labels=None) -> pd.Series:
    """Dispatch to equal-width or equal-frequency binning."""
    method = method.lower()
    if method in ("equal_width", "width", "ew"):
        return equal_width_binning(series, n_bins=n_bins, labels=labels)
    elif method in ("equal_freq", "frequency", "ef", "quantile"):
        return equal_freq_binning(series, n_bins=n_bins, labels=labels)
    else:
        raise ValueError(f"Unknown binning method: {method}")

def main():
    parser = argparse.ArgumentParser(description="A4: Flexible binning for a chosen column.")
    parser.add_argument("--csv", default="DCT_mal.csv", help="Path to CSV file.")
    parser.add_argument("--column", default=None, help="Column to bin. If omitted, uses last column.")
    parser.add_argument("--bins", type=int, default=4, help="Number of bins.")
    parser.add_argument("--method", default="equal_width", choices=["equal_width", "equal_freq"], help="Binning method.")
    parser.add_argument("--out", default="A4_binned_preview.csv", help="Where to save a small preview CSV.")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    col = args.column or df.columns[-1]
    binned = bin_series(df[col], method=args.method, n_bins=args.bins)
    df_out = df.copy()
    df_out[col + f"_{args.method}_{args.bins}"] = binned.astype(str)

    # Save a short preview (first 50 rows) to file
    df_out.head(50).to_csv(args.out, index=False)
    print(f"Saved preview with binned column to {args.out}")

if __name__ == "__main__":
    main()

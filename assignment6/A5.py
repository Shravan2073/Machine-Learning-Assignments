
import argparse
import numpy as np
import pandas as pd
from collections import Counter, defaultdict

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
    y = y.astype(str)
    x = x.astype(str)
    H_y = entropy_from_counts(Counter(y).values())
    total = len(y)
    cond = 0.0
    for v, idx in x.groupby(x).groups.items():
        y_subset = y.loc[idx]
        cond += (len(y_subset) / total) * entropy_from_counts(Counter(y_subset).values())
    return H_y - cond

class DTNode:
    def __init__(self, prediction=None, feature=None, children=None):
        self.prediction = prediction   # class label if leaf
        self.feature = feature         # feature to split on (categorical str feature)
        self.children = children or {} # dict value -> DTNode

    def is_leaf(self):
        return self.feature is None

class SimpleDecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        self.features_ = None

    def _majority_class(self, y):
        c = Counter(y.astype(str))
        return max(c, key=c.get)

    def _build(self, X: pd.DataFrame, y: pd.Series, depth: int):
        # stopping conditions
        y_str = y.astype(str)
        if len(set(y_str)) == 1:
            return DTNode(prediction=y_str.iloc[0])
        if self.max_depth is not None and depth >= self.max_depth:
            return DTNode(prediction=self._majority_class(y_str))
        if len(X) < self.min_samples_split or X.shape[1] == 0:
            return DTNode(prediction=self._majority_class(y_str))

        # choose best feature by IG
        best_feat, best_ig = None, -1.0
        for feat in X.columns:
            ig = information_gain(y_str, X[feat].astype(str))
            if ig > best_ig:
                best_ig, best_feat = ig, feat

        if best_feat is None or best_ig <= 0:
            return DTNode(prediction=self._majority_class(y_str))

        node = DTNode(feature=best_feat)
        for v, idx in X[best_feat].astype(str).groupby(X[best_feat].astype(str)).groups.items():
            X_sub = X.loc[idx].drop(columns=[best_feat])
            y_sub = y.loc[idx]
            child = self._build(X_sub, y_sub, depth+1)
            node.children[v] = child
        return node

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.features_ = list(X.columns)
        self.root = self._build(X, y, depth=0)
        return self

    def _predict_row(self, row, node: 'DTNode'):
        if node.is_leaf():
            return node.prediction
        v = str(row[node.feature])
        child = node.children.get(v, None)
        if child is None:
            # unseen category -> fallback to majority among children
            counts = Counter(ch.prediction for ch in node.children.values() if ch.is_leaf())
            if counts:
                return max(counts, key=counts.get)
            # otherwise any child predict
            return next(iter(node.children.values())).prediction
        return self._predict_row(row, child)

    def predict(self, X: pd.DataFrame):
        return np.array([self._predict_row(r, self.root) for _, r in X.iterrows()])

def preprocess_to_categorical(df: pd.DataFrame, target: str, binning: str = "equal_width", bins: int = 4):
    X = df.drop(columns=[target]).copy()
    y = df[target].copy()
    # bin numeric target for classification if numeric
    if np.issubdtype(y.dropna().dtype, np.number):
        y = equal_width_binning(y, n_bins=bins)

    for c in X.columns:
        col = X[c]
        if np.issubdtype(col.dropna().dtype, np.number):
            if binning == "equal_width":
                X[c] = equal_width_binning(col, n_bins=bins)
            else:
                X[c] = equal_freq_binning(col, n_bins=bins)
        else:
            X[c] = col.astype(str)
    return X.astype(str), y.astype(str)

def main():
    parser = argparse.ArgumentParser(description="A5: Build your own Decision Tree (ID3-like) with binning.")
    parser.add_argument("--csv", default="DCT_mal.csv")
    parser.add_argument("--target", default=None, help="Target column (defaults to last).")
    parser.add_argument("--binning", default="equal_width", choices=["equal_width", "equal_freq"])
    parser.add_argument("--bins", type=int, default=4)
    parser.add_argument("--max_depth", type=int, default=None)
    parser.add_argument("--min_samples_split", type=int, default=2)
    parser.add_argument("--report", default="A5_results.txt")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    target = args.target or df.columns[-1]

    X_cat, y_cat = preprocess_to_categorical(df, target, binning=args.binning, bins=args.bins)
    tree = SimpleDecisionTree(max_depth=args.max_depth, min_samples_split=args.min_samples_split)
    tree.fit(X_cat, y_cat)

    # simple resubstitution accuracy
    preds = tree.predict(X_cat)
    acc = (preds == y_cat.values).mean()

    with open(args.report, "w") as f:
        f.write(f"Training (resub) accuracy: {acc:.4f}\n")
        f.write(f"Features used: {list(X_cat.columns)}\n")
    print(f"Saved results to {args.report}")

if __name__ == "__main__":
    main()

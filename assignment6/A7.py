
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier

def encode(X2: pd.DataFrame):
    X2 = X2.copy()
    for c in X2.columns:
        if not np.issubdtype(X2[c].dropna().dtype, np.number):
            enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            X2[[c]] = enc.fit_transform(X2[[c]].astype(str))
    return X2

def pick_two_features(X: pd.DataFrame):
    # heuristic: first two numeric, else first two columns
    num_cols = [c for c in X.columns if np.issubdtype(X[c].dropna().dtype, np.number)]
    if len(num_cols) >= 2:
        return num_cols[:2]
    return X.columns[:2]

def main():
    parser = argparse.ArgumentParser(description="A7: Visualize DT decision boundary for 2 features.")
    parser.add_argument("--csv", default="DCT_mal.csv")
    parser.add_argument("--target", default=None, help="Target column (defaults to last).")
    parser.add_argument("--features", nargs=2, default=None, help="Two feature names to use.")
    parser.add_argument("--max_depth", type=int, default=4)
    parser.add_argument("--out", default="A7_decision_boundary.png")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    target = args.target or df.columns[-1]
    X = df.drop(columns=[target])
    y = df[target]

    # If y is numeric with many unique levels -> bin to 3 classes for visualization
    if np.issubdtype(y.dropna().dtype, np.number) and y.nunique() > 6:
        y = pd.qcut(y, q=3, labels=False, duplicates="drop")

    if args.features:
        f1, f2 = args.features
    else:
        f1, f2 = pick_two_features(X)

    X2 = X[[f1, f2]].copy()
    X2_enc = encode(X2)

    X_train, X_test, y_train, y_test = train_test_split(X2_enc, y, test_size=0.3, random_state=42, stratify=y if pd.Series(y).nunique() > 1 else None)

    clf = DecisionTreeClassifier(max_depth=args.max_depth, random_state=42)
    clf.fit(X_train, y_train)

    # Create mesh grid
    x_min, x_max = X2_enc[f1].min() - 1, X2_enc[f1].max() + 1
    y_min, y_max = X2_enc[f2].min() - 1, X2_enc[f2].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
    grid = pd.DataFrame({f1: xx.ravel(), f2: yy.ravel()})
    Z = clf.predict(grid).reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3)
    # Plot training points
    plt.scatter(X_train[f1], X_train[f2], s=20, alpha=0.9)
    plt.xlabel(f1)
    plt.ylabel(f2)
    plt.title("Decision Tree Decision Boundary (2 features)")
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"Saved decision boundary to {args.out}")

if __name__ == "__main__":
    main()

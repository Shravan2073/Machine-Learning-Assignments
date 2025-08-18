import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

def auto_split_X_y(df: pd.DataFrame, target: str):
    X = df.drop(columns=[target])
    y = df[target]
    return X, y

def encode_features(X: pd.DataFrame):
    X_enc = X.copy()
    cat_cols = [c for c in X.columns if not np.issubdtype(X[c].dropna().dtype, np.number)]
    if cat_cols:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X_enc[cat_cols] = enc.fit_transform(X[cat_cols].astype(str))
    return X_enc

def main():
    parser = argparse.ArgumentParser(description="A6: Train & visualize a Decision Tree on DCT_mal.csv")
    parser.add_argument("--csv", default="DCT_mal.csv")
    parser.add_argument("--target", default=None, help="Target column (defaults to last).")
    parser.add_argument("--max_depth", type=int, default=4)
    parser.add_argument("--out", default="A6_tree.png")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    target = args.target or df.columns[-1]
    X, y = auto_split_X_y(df, target)

    # If y is numeric with many unique values, bin it to classification
    if np.issubdtype(y.dropna().dtype, np.number) and y.nunique() > 10:
        # bin into 4 classes
        y = pd.qcut(y, q=4, labels=False, duplicates="drop")

    X_enc = encode_features(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_enc, y, test_size=0.3, random_state=42,
        stratify=y if pd.Series(y).nunique() > 1 else None
    )

    clf = DecisionTreeClassifier(max_depth=args.max_depth, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {acc:.4f}")

    # High-resolution figure
    plt.figure(figsize=(20, 14), dpi=300)
    plot_tree(
        clf,
        feature_names=list(X_enc.columns),
        class_names=[str(c) for c in sorted(pd.Series(y).unique())],
        filled=False,
        fontsize=10
    )
    plt.tight_layout()
    plt.savefig(args.out, dpi=300)
    print(f"Saved tree visualization to {args.out}")

if __name__ == "__main__":
    main()

"""
A1: Linear regression using one attribute (feature) and target.
Follows Lab Session 05 coding rules:
 - All functionality is in functions (no prints inside functions).
 - main() runs the flow and prints results.
 - Adjust TARGET_COL if needed.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

CSV_PATH = "DCT_mal.csv"
TARGET_COL = None  # None -> auto-detect (prefers 'target'/'class' or last column)
RANDOM_STATE = 42
TEST_SIZE = 0.2

def load_and_prepare(path, target_col=None):
    df = pd.read_csv(path)
    # Drop rows with NaNs for simplicity
    df = df.dropna().reset_index(drop=True)
    if target_col is None:
        for name in ["target", "y", "label", "class"]:
            if name in df.columns:
                target_col = name
                break
    if target_col is None:
        target_col = df.columns[-1]  # fallback: last column
    # pick a single numeric feature (first numeric column that is not target)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    if not numeric_cols:
        raise ValueError("No numeric feature available besides target.")
    feature_col = numeric_cols[0]
    X = df[[feature_col]].values
    y = df[target_col].values
    return X, y, feature_col, target_col

def train_linear_regression(X_train, y_train):
    reg = LinearRegression().fit(X_train, y_train)
    return reg

def main():
    X, y, feat, target = load_and_prepare(CSV_PATH, TARGET_COL)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    reg = train_linear_regression(X_train, y_train)
    y_train_pred = reg.predict(X_train)
    y_test_pred = reg.predict(X_test)

    # print short summary
    print(f"Used feature: {feat}, target: {target}")
    print("Model coefficients:", reg.coef_)
    print("Model intercept:", reg.intercept_)
    # leave metrics to A2 script

if __name__ == "__main__":
    main()

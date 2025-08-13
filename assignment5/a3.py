"""
A3: Repeat linear regression with multiple features (all numeric features except target).
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

CSV_PATH = "DCT_mal.csv"
TARGET_COL = None
TEST_SIZE = 0.3
RANDOM_STATE = 42

def load_all_numeric_features(path, target_col=None):
    df = pd.read_csv(path).dropna().reset_index(drop=True)
    if target_col is None:
        for name in ["target", "y", "label", "class"]:
            if name in df.columns:
                target_col = name
                break
    if target_col is None:
        target_col = df.columns[-1]
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    if not numeric_cols:
        raise ValueError("No numeric features found.")
    X = df[numeric_cols].values
    y = df[target_col].values
    return X, y, numeric_cols, target_col

def compute_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    denom = np.where(y_true == 0, 1e-8, y_true)
    mape = np.mean(np.abs((y_true - y_pred) / denom)) * 100
    r2 = r2_score(y_true, y_pred)
    return {"MSE": mse, "RMSE": rmse, "MAPE(%)": mape, "R2": r2}

def train_and_evaluate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    reg = LinearRegression().fit(X_train, y_train)
    y_train_pred = reg.predict(X_train)
    y_test_pred = reg.predict(X_test)
    return reg, compute_metrics(y_train, y_train_pred), compute_metrics(y_test, y_test_pred)

def main():
    X, y, feature_names, target = load_all_numeric_features(CSV_PATH, TARGET_COL)
    reg, metrics_train, metrics_test = train_and_evaluate(X, y)
    print(f"Using features: {feature_names}")
    print("Train metrics:")
    for k, v in metrics_train.items():
        print(f"  {k}: {v}")
    print("Test metrics:")
    for k, v in metrics_test.items():
        print(f"  {k}: {v}")
    print("Regression coefficients (aligned to features):")
    print(list(zip(feature_names, reg.coef_)))
    print("Intercept:", reg.intercept_)

if __name__ == "__main__":
    main()

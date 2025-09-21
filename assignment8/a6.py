import pandas as pd
import numpy as np

def sigmoid(x): return 1 / (1 + np.exp(-x))

def perceptron_train(X, y, lr=0.05, epochs=500):
    w = np.zeros(X.shape[1] + 1)
    for _ in range(epochs):
        for xi, target in zip(X, y):
            xi_aug = np.insert(xi, 0, 1)
            net = np.dot(w, xi_aug)
            out = sigmoid(net)
            err = target - out
            w += lr * err * xi_aug
    return w

def main():
    df = pd.read_csv("DCT_mal.csv")  # project dataset
    # Example: use first 3 features and binary target column (adjust as per your dataset)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].apply(lambda v: 1 if v in [1, "Yes", "High"] else 0).values

    w = perceptron_train(X, y)
    print("Final Weights:", w)

if __name__ == "__main__":
    main()

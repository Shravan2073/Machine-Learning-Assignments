import numpy as np

def step_activation(x): return 1 if x >= 0 else 0

def perceptron_train_multi(X, y, lr=0.05, epochs=1000):
    w = np.zeros((X.shape[1] + 1, y.shape[1]))
    for _ in range(epochs):
        for xi, target in zip(X, y):
            xi_aug = np.insert(xi, 0, 1)
            net = xi_aug.dot(w)
            out = np.array([step_activation(v) for v in net])
            err = target - out
            w += lr * np.outer(xi_aug, err)
    return w

def main():
    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    y = np.array([[1,0],[1,0],[1,0],[0,1]])  # multi-output encoding

    w = perceptron_train_multi(X, y)
    print("Final Weights:\n", w)

if __name__ == "__main__":
    main()

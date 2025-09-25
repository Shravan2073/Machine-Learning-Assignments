import numpy as np

def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

def perceptron_train(X, y, lr=0.05, epochs=1000):
    w = np.random.randn(X.shape[1] + 1) * 0.01
    for _ in range(epochs):
        for xi, target in zip(X, y):
            xi_aug = np.insert(xi, 0, 1)
            net = np.dot(w, xi_aug)
            out = sigmoid(net)
            err = target - out
            w += lr * err * xi_aug
    return w

def main():
    X = np.array([
        [20,6,2,386],
        [16,3,6,289],
        [27,6,2,393],
        [19,1,2,110],
        [24,4,2,280],
        [22,1,5,167],
        [15,4,2,271],
        [18,4,2,274],
        [21,1,4,148],
        [16,2,4,198]
    ])
    y = np.array([1,1,1,0,1,0,1,1,0,0])  # Yes=1, No=0

    # Train perceptron
    w_perc = perceptron_train(X, y)
    print("Perceptron Weights:", w_perc)

    # Pseudo-inverse solution
    X_aug = np.c_[np.ones(X.shape[0]), X]
    w_pinv = np.linalg.pinv(X_aug).dot(y)
    print("Pseudo-inverse Weights:", w_pinv)

if __name__ == "__main__":
    main()

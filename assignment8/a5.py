import numpy as np
import matplotlib.pyplot as plt

def step_activation(x): return 1 if x >= 0 else 0

def perceptron_train(X, y, lr=0.05, max_epochs=1000):
    w = np.array([10, 0.2, -0.75], dtype=float)
    errors = []
    for epoch in range(max_epochs):
        total_error = 0
        for xi, target in zip(X, y):
            xi_aug = np.insert(xi, 0, 1)
            net = np.dot(w, xi_aug)
            out = step_activation(net)
            err = target - out
            w += lr * err * xi_aug
            total_error += err**2
        errors.append(total_error)
    return w, errors

def main():
    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    y = np.array([0,1,1,0])  # XOR gate

    w, errors = perceptron_train(X, y)
    print("Final Weights:", w)

    plt.plot(errors)
    plt.title("A5: XOR Gate Error Convergence (Perceptron)")
    plt.xlabel("Epoch")
    plt.ylabel("SSE")
    plt.show()

if __name__ == "__main__":
    main()

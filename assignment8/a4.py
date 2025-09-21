import numpy as np
import matplotlib.pyplot as plt

def step_activation(x): return 1 if x >= 0 else 0

def perceptron_train(X, y, lr=0.1, max_epochs=1000, tol=0.002):
    w = np.array([10, 0.2, -0.75], dtype=float)
    for epoch in range(max_epochs):
        total_error = 0
        for xi, target in zip(X, y):
            xi_aug = np.insert(xi, 0, 1)
            net = np.dot(w, xi_aug)
            out = step_activation(net)
            err = target - out
            w += lr * err * xi_aug
            total_error += err**2
        if total_error <= tol:
            break
    return epoch + 1

def main():
    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    y = np.array([0,0,0,1])  # AND gate

    rates = np.arange(0.1, 1.1, 0.1)
    epochs_list = []

    for lr in rates:
        epochs = perceptron_train(X, y, lr=lr)
        epochs_list.append(epochs)
        print(f"LR={lr:.1f} → Epochs={epochs}")

    plt.plot(rates, epochs_list, marker="o")
    plt.title("A4: Learning Rate vs Epochs (AND Gate)")
    plt.xlabel("Learning Rate")
    plt.ylabel("Epochs to Converge")
    plt.show()

if __name__ == "__main__":
    main()

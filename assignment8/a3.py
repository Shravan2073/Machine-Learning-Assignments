import numpy as np

def bipolar_step_activation(x): return 1 if x >= 0 else -1
def sigmoid_activation(x): return 1 / (1 + np.exp(-x))
def relu_activation(x): return max(0, x)
def step_from_fn(fn, x): return 1 if fn(x) >= 0.5 else 0

def perceptron_train(X, y, activation_fn, lr=0.05, max_epochs=1000, tol=0.002):
    w = np.array([10, 0.2, -0.75], dtype=float)
    for epoch in range(max_epochs):
        total_error = 0
        for xi, target in zip(X, y):
            xi_aug = np.insert(xi, 0, 1)
            net = np.dot(w, xi_aug)
            out = activation_fn(net)
            err = target - out
            w += lr * err * xi_aug
            total_error += err**2
        if total_error <= tol:
            break
    return epoch + 1

def main():
    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    y = np.array([0,0,0,1])  # AND gate

    activations = {
        "Bipolar Step": lambda x: 1 if bipolar_step_activation(x) == 1 else 0,
        "Sigmoid": lambda x: step_from_fn(sigmoid_activation, x),
        "ReLU": lambda x: 1 if relu_activation(x) > 0 else 0
    }

    for name, fn in activations.items():
        epochs = perceptron_train(X, y, fn)
        print(f"{name} converged in {epochs} epochs")

if __name__ == "__main__":
    main()

import numpy as np

def sigmoid(x): return 1 / (1 + np.exp(-x))
def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def train_backprop(X, y, lr=0.05, epochs=1000, tol=0.002):
    np.random.seed(0)
    W1 = np.random.randn(2, 2) * 0.1
    W2 = np.random.randn(2, 1) * 0.1

    for epoch in range(epochs):
        z1 = X.dot(W1)
        a1 = sigmoid(z1)
        z2 = a1.dot(W2)
        a2 = sigmoid(z2)

        error = y - a2
        if np.mean(error**2) < tol:
            break

        d2 = error * sigmoid_derivative(z2)
        d1 = d2.dot(W2.T) * sigmoid_derivative(z1)

        W2 += a1.T.dot(d2) * lr
        W1 += X.T.dot(d1) * lr

    return W1, W2, epoch + 1

def main():
    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    y = np.array([[0],[1],[1],[0]])  # XOR gate

    W1, W2, epochs = train_backprop(X, y)
    print("Trained in epochs:", epochs)
    print("W1:", W1)
    print("W2:", W2)

if __name__ == "__main__":
    main()

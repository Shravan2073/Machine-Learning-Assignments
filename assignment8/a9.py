import numpy as np
from A8 import train_backprop  # reuse function

def main():
    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    y = np.array([[0],[1],[1],[0]])  # XOR gate

    W1, W2, epochs = train_backprop(X, y)
    print("Trained in epochs:", epochs)
    print("W1:", W1)
    print("W2:", W2)

if __name__ == "__main__":
    main()

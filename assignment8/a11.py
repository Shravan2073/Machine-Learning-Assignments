import numpy as np
from sklearn.neural_network import MLPClassifier

def main():
    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    y_and = np.array([0,0,0,1])
    y_xor = np.array([0,1,1,0])

    clf_and = MLPClassifier(hidden_layer_sizes=(2,), max_iter=1000)
    clf_and.fit(X, y_and)
    print("AND Gate Predictions:", clf_and.predict(X))

    clf_xor = MLPClassifier(hidden_layer_sizes=(2,), max_iter=1000)
    clf_xor.fit(X, y_xor)
    print("XOR Gate Predictions:", clf_xor.predict(X))

if __name__ == "__main__":
    main()

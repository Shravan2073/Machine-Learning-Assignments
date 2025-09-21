import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

def main():
    df = pd.read_csv("DCT_mal.csv")
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].apply(lambda v: 1 if v in [1, "Yes", "High"] else 0).values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = MLPClassifier(hidden_layer_sizes=(10,), activation="relu", max_iter=1000)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print("MLP Accuracy on Project Dataset:", acc)

if __name__ == "__main__":
    main()

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

def load_thyroid_data(filename):
    df = pd.read_excel(filename, sheet_name='thyroid0387_UCI')
    df = df.replace('?', pd.NA)
    df = df.dropna(subset=["age", "TSH", "Condition"])
    df["age"] = pd.to_numeric(df["age"], errors='coerce')
    df["TSH"] = pd.to_numeric(df["TSH"], errors='coerce')
    df = df.dropna(subset=["age", "TSH"])
    df["target"] = (df["Condition"] != "NO CONDITION").astype(int)
    X = df[["age", "TSH"]]
    y = df["target"]
    return X, y

def train_knn_classifier(X, y, k=3):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    return model, X_train, X_test, y_train, y_test

def evaluate_classification(model, X, y_true):
    y_pred = model.predict(X)
    cm = confusion_matrix(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return cm, precision, recall, f1

if __name__ == "__main__":
    X, y = load_thyroid_data("temp_DATASET.xlsx")
    knn_model, X_train, X_test, y_train, y_test = train_knn_classifier(X, y)
    cm_train, prec_train, rec_train, f1_train = evaluate_classification(knn_model, X_train, y_train)
    cm_test, prec_test, rec_test, f1_test = evaluate_classification(knn_model, X_test, y_test)
    print("TRAIN Confusion matrix:\n", cm_train)
    print("TRAIN Precision:", prec_train, "Recall:", rec_train, "F1:", f1_train)
    print("TEST Confusion matrix:\n", cm_test)
    print("TEST Precision:", prec_test, "Recall:", rec_test, "F1:", f1_test)
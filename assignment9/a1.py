import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import StackingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def load_data(path="DCT_mal.csv"):
    df = pd.read_csv(path)
    X = df.iloc[:, :-1]   # all columns except last as features
    y = df.iloc[:, -1]    # last column as label
    return X, y

def build_stacking_classifier(X_train, y_train):
    # Base models
    base_estimators = [
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('gb', GradientBoostingClassifier(random_state=42)),
        ('svc', SVC(probability=True, random_state=42))
    ]

    # Meta model (final estimator)
    final_estimator = LogisticRegression()

    # Stacking Classifier
    stack_clf = StackingClassifier(
        estimators=base_estimators,
        final_estimator=final_estimator,
        cv=5
    )

    stack_clf.fit(X_train, y_train)
    return stack_clf

if __name__ == "__main__":
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Build & train stacking classifier
    model = build_stacking_classifier(X_train_scaled, y_train)

    # Predictions
    y_pred = model.predict(X_test_scaled)

    # Accuracy
    print("Stacking Classifier Accuracy:", accuracy_score(y_test, y_pred))

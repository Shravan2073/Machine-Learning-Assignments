import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def load_data(path="DCT_mal.csv"):
    df = pd.read_csv(path)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    return X, y

def build_pipeline():
    pipeline = Pipeline([
        ('scaler', StandardScaler()),  # Step 1: Scaling
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))  # Step 2: Model
    ])
    return pipeline

if __name__ == "__main__":
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build and train pipeline
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    # Predictions
    y_pred = pipeline.predict(X_test)

    # Accuracy
    print("Pipeline Accuracy:", accuracy_score(y_test, y_pred))

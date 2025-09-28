import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from lime.lime_tabular import LimeTabularExplainer

def load_data(path="DCT_mal.csv"):
    df = pd.read_csv(path)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    return X, y

def build_pipeline():
    pipeline = Pipeline([
        ('scaler', StandardScaler()),  
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))  
    ])
    return pipeline

if __name__ == "__main__":
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build and train pipeline
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    # Create LIME explainer
    explainer = LimeTabularExplainer(
        training_data=X_train,
        feature_names=[f"Feature_{i}" for i in range(X.shape[1])],
        class_names=list(map(str, set(y))),
        discretize_continuous=True
    )

    # Pick one test instance for explanation
    instance = X_test[0].reshape(1, -1)
    exp = explainer.explain_instance(
        data_row=X_test[0],
        predict_fn=pipeline.predict_proba,
        num_features=5
    )

    print("Explaining instance:", X_test[0])
    print("Prediction:", pipeline.predict(instance))
    exp.show_in_notebook(show_table=True)   # for Jupyter
    exp.save_to_file("lime_explanation.html")  # for offline viewing

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def pca_classification_95(filepath):
    """
    Performs PCA retaining 95% variance and runs a classification model.

    Args:
        filepath (str): The path to the CSV file.
    """
    df = pd.read_csv(filepath)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply PCA
    pca = PCA(n_components=0.95)
    X_pca = pca.fit_transform(X_scaled)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X_pca, y, test_size=0.2, random_state=42
    )

    # Train a Logistic Regression model
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Number of components selected: {pca.n_components_}")
    print(f"Accuracy with 95% variance: {accuracy:.4f}")

# Run the function for A3
pca_classification_95('DCT_mal.csv')
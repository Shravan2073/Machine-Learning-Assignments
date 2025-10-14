import pandas as pd
import lime
import lime.lime_tabular
import shap
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def explain_model_with_lime_and_shap(filepath):
    """
    Explains a model's behavior using LIME and SHAP.

    Args:
        filepath (str): The path to the CSV file.
    """
    df = pd.read_csv(filepath)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train a RandomForestClassifier
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # LIME Explainer
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=X_train.columns.tolist(),
        class_names=['benign', 'malicious'],
        mode='classification'
    )

    # Explain a single instance
    i = 0
    exp = explainer.explain_instance(
        data_row=X_test.iloc[i].values,
        predict_fn=model.predict_proba
    )

    print(f"LIME Explanation for instance {i}:")
    print(exp.as_list())

    # SHAP Explainer
    shap_explainer = shap.TreeExplainer(model)
    shap_values = shap_explainer.shap_values(X_test)

    print("\nSHAP Summary Plot:")
    shap.summary_plot(shap_values, X_test)

# Run the function for A5
explain_model_with_lime_and_shap('DCT_mal.csv')
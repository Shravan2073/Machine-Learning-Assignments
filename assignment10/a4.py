import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SequentialFeatureSelector
import time # Import time to measure execution duration

def sequential_feature_selection_classification_optimized(filepath):
    """
    Identifies the best subset of features using a fast and efficient
    Sequential Feature Selector and evaluates a classification model.

    Args:
        filepath (str): The path to the CSV file ('DCT_mal.csv').
    """
    print("Starting A4: Sequential Feature Selection...")
    start_time = time.time()

    # 1. Load the dataset
    df = pd.read_csv(filepath)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # 2. Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3. Initialize the model that will be used for selection
    # Using n_jobs=-1 tells the model to use all available CPU cores, speeding it up.
    model = RandomForestClassifier(n_jobs=-1, random_state=42)

    # 4. Set up the Sequential Feature Selector (SFS)
    # This is the core of the method.
    sfs = SequentialFeatureSelector(
        model,
        n_features_to_select=15,  # Key change: We explicitly ask for the best 15 features.
        direction='forward',      # Start with 0 features and add one by one.
        cv=2,                     # Use 2-fold cross-validation to evaluate features.
        n_jobs=-1                 # Use all available CPU cores for cross-validation.
    )

    # 5. Run the feature selection process
    print("Fitting the Sequential Feature Selector... (This may take a few minutes)")
    sfs.fit(X_train, y_train)
    
    # 6. Get the names of the selected features
    selected_features = list(X_train.columns[sfs.get_support()])

    # 7. Create new training and testing sets with only the selected features
    X_train_sfs = sfs.transform(X_train)
    X_test_sfs = sfs.transform(X_test)

    # 8. Train the final model on the reduced feature set
    print("Training the final model on selected features...")
    model.fit(X_train_sfs, y_train)

    # 9. Evaluate the model's performance
    y_pred = model.predict(X_test_sfs)
    accuracy = accuracy_score(y_test, y_pred)
    
    end_time = time.time()
    duration = end_time - start_time

    # 10. Print the final results
    print("\n--- SFS Results ---")
    print(f"Process completed in {duration:.2f} seconds.")
    print(f"Number of features selected: {len(selected_features)}")
    print(f"Selected features are: {selected_features}")
    print(f"Final Model Accuracy: {accuracy:.4f}")

# --- Main execution ---
if __name__ == '__main__':
    filepath = 'DCT_mal.csv'
    sequential_feature_selection_classification_optimized(filepath)
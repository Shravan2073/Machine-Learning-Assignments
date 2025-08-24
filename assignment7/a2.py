# a2_hyperparameter_tuning.py (Corrected)

import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import randint
from sklearn.preprocessing import LabelEncoder # <-- IMPORT LabelEncoder

# Load the dataset
try:
    data = pd.read_csv('DCT_mal.csv')
except FileNotFoundError:
    print("Error: DCT_mal.csv not found. Please ensure the file is in the same directory.")
    exit()

# Separate features (X) and target label (y)
X = data.drop('LABEL', axis=1)
y = data['LABEL']

# --- FIX: Encode the labels ---
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Define the model
dt_classifier = DecisionTreeClassifier(random_state=42)

# Define the hyperparameter distribution
param_dist = {
    'max_depth': randint(3, 25),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 20),
    'criterion': ['gini', 'entropy']
}

# Instantiate RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=dt_classifier,
    param_distributions=param_dist,
    n_iter=15,
    cv=5,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

# Fit the model
print("🚀 Starting hyperparameter tuning...")
random_search.fit(X_train, y_train)

# Print the best parameters and score
print("\n✅ Best Parameters found: ", random_search.best_params_)
print(f"📈 Best Cross-Validation Accuracy: {random_search.best_score_:.4f}")

# Evaluate the best model on the test set
best_classifier = random_search.best_estimator_
accuracy = best_classifier.score(X_test, y_test)
print(f"📊 Test Set Accuracy with Best Parameters: {accuracy:.4f}")
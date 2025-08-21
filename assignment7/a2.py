import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("DCT_mal.csv")

X = df.drop(columns=["LABEL"])
y = df["LABEL"]

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Search space
param_dist = {
    "criterion": ["gini", "entropy", "log_loss"],
    "max_depth": [None, 5, 10, 20, 50],
    "min_samples_split": [2, 5, 10, 20],
    "min_samples_leaf": [1, 2, 4, 10]
}

clf = DecisionTreeClassifier(random_state=42)

# Randomized search
random_search = RandomizedSearchCV(
    clf, param_distributions=param_dist, n_iter=20, cv=5,
    scoring="accuracy", random_state=42, n_jobs=-1
)
random_search.fit(X_train, y_train)

best_model = random_search.best_estimator_

print("Best Parameters:", random_search.best_params_)
print("Best CV Accuracy:", random_search.best_score_)
print("Test Accuracy:", accuracy_score(y_test, best_model.predict(X_test)))

# a3.py (Corrected)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder # <-- 1. IMPORT LabelEncoder

# Load the dataset
try:
    data = pd.read_csv('DCT_mal.csv')
except FileNotFoundError:
    print("Error: DCT_mal.csv not found. Please ensure the file is in the same directory.")
    exit()

# Separate features (X) and target label (y)
X = data.drop('LABEL', axis=1)
y = data['LABEL']

# --- FIX: Encode the labels into a 0-indexed format ---
le = LabelEncoder() # <-- 2. CREATE an instance of the encoder
y_encoded = le.fit_transform(y) # <-- 3. FIT the encoder and TRANSFORM the labels

# Split data using the new encoded labels
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Scale features for MLP
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 1. XGBoost Classifier ---
print("--- Training XGBoost Classifier ---")
# No 'use_label_encoder=False' needed now, but keeping it is fine
xgb_classifier = XGBClassifier(random_state=42, eval_metric='mlogloss')
xgb_classifier.fit(X_train, y_train)
y_pred_xgb = xgb_classifier.predict(X_test)
print("\nPerformance Report for XGBoost:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_xgb):.4f}")
print(classification_report(y_test, y_pred_xgb, zero_division=0))

# --- 2. Gaussian Naive Bayes Classifier ---
print("\n--- Training Gaussian Naive Bayes Classifier ---")
gnb_classifier = GaussianNB()
gnb_classifier.fit(X_train, y_train)
y_pred_gnb = gnb_classifier.predict(X_test)
print("\nPerformance Report for Gaussian Naive Bayes:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_gnb):.4f}")
print(classification_report(y_test, y_pred_gnb, zero_division=0))

# --- 3. MLP (Multi-layer Perceptron) Classifier ---
print("\n--- Training MLP Classifier ---")
mlp_classifier = MLPClassifier(random_state=42, max_iter=1000, hidden_layer_sizes=(100, 50))
mlp_classifier.fit(X_train_scaled, y_train)
y_pred_mlp = mlp_classifier.predict(X_test_scaled)
print("\nPerformance Report for MLP Classifier:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_mlp):.4f}")
print(classification_report(y_test, y_pred_mlp, zero_division=0))
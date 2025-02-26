import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier  # Use your model (e.g., XGBoost, LightGBM)

# Load the PCOS dataset
df = pd.read_csv("PCOS_data.csv")

# Preprocess dataset (ensure correct column selection)
X = df.drop(columns=["PCOS"])  # Replace "PCOS" with the correct target column
y = df["PCOS"]  # Target column

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, "pcos_model.pkl")

print("✅ Model saved successfully as 'pcos_model.pkl'")

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("PCOS_data.csv")

# Ensure correct target column
print("Columns in dataset:", df.columns)

target_column = "PCOS"  # Change if your dataset has a different target column name
if target_column not in df.columns:
    raise ValueError(f"Error: Target column '{target_column}' not found in dataset.")

# Select features and target
X = df[["Age", "BMI"]]  # Adjust based on dataset
y = df[target_column]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Save model
joblib.dump(model, "pcos_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("✅ Model training complete! pcos_model.pkl saved.")

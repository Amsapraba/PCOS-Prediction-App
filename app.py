import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

df = pd.read_csv("PCOS_data.csv", nrows=500)  


df.drop(columns=["Sl. No", "Patient File No.", "Unnamed: 44"], errors='ignore', inplace=True)


for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')  


df.dropna(inplace=True)


categorical_columns = ['Blood Group', 'Fast food (Y/N)', 'Reg.Exercise(Y/N)']
for col in categorical_columns:
    if col in df.columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))


X = df.drop(columns=['PCOS (Y/N)'], errors='ignore')  
y = df['PCOS (Y/N)']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)


y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Model Accuracy: {accuracy:.2f}")
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)


feature_importances = pd.Series(model.feature_importances_, index=X.columns)
feature_importances.sort_values(ascending=False).plot(kind='bar', figsize=(12, 6))
plt.title("Feature Importance")
plt.show()


model_filename = "pcos_random_forest_model.pkl"
scaler_filename = "scaler.pkl"
joblib.dump(model, model_filename)
joblib.dump(scaler, scaler_filename)

print(f"Model saved as {model_filename}")
print(f"Scaler saved as {scaler_filename}")


print(f"Number of features in training data: {X.shape[1]}")
print("Feature names:", X.columns.tolist())


sample_input = np.array([[
    28, 19.3, 44.6, 152, 78, 10.48, 22, 5, 7, 0, 0, 15, 1, 0,
    1.99, 1.99, 7.95, 3.68, 2.16, 36, 30, 0.83, 0.68, 2.07,
    45.16, 17.1, 0.57, 92, 0, 0, 0, 0, 0, 1, 0, 110, 80, 3,
    3, 18, 18
]])  


sample_input_scaled = scaler.transform(sample_input)
prediction = model.predict(sample_input_scaled)
print(f"Predicted PCOS (Y/N): {prediction[0]}")

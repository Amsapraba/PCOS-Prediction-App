import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import streamlit as st

# Load Dataset
df = pd.read_csv("PCOS_data.csv")

# Clean column names
df.columns = df.columns.str.strip().str.replace(" ", "_").str.replace("/", "_")

# Drop unnecessary columns
df.drop(columns=["Sl_No", "Patient_File_No", "Unnamed_44"], inplace=True, errors='ignore')

# Convert object columns to numeric
df["II_beta_HCG_mIU_mL"] = pd.to_numeric(df["II_beta_HCG_mIU_mL"], errors='coerce')
df["AMH_ng_mL"] = pd.to_numeric(df["AMH_ng_mL"], errors='coerce')
df["Avg_F_size_R_mm"] = pd.to_numeric(df["Avg_F_size_R_mm"], errors='coerce')

# Fill missing values
df.fillna(df.median(), inplace=True)

# Define features and target
X = df.drop(columns=["PCOS_Y_N"])
y = df["PCOS_Y_N"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define Models
rf = RandomForestClassifier(n_estimators=100, random_state=42)
xgb = XGBClassifier(eval_metric='logloss')
lr = LogisticRegression()

# Stacking Ensemble
stacking_model = StackingClassifier(estimators=[('rf', rf), ('xgb', xgb)], final_estimator=lr)
stacking_model.fit(X_train, y_train)

# Predictions
y_pred = stacking_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Streamlit App
def predict_pcos(user_input):
    user_input = np.array(user_input).reshape(1, -1)
    user_input = scaler.transform(user_input)
    prediction = stacking_model.predict(user_input)
    return "PCOS Detected" if prediction[0] == 1 else "No PCOS"

st.title("PCOS Prediction App")
st.write("Enter details to predict PCOS")

# User Inputs
user_data = [st.number_input(col, value=0.0) for col in X.columns]

if st.button("Predict"):
    result = predict_pcos(user_data)
    st.write(result)

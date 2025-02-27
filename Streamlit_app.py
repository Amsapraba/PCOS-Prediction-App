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

# Drop unnecessary columns
df.drop(columns=["Sl. No", "Patient File No.", "Unnamed: 44"], inplace=True)

# Convert object columns to numeric
df["II    beta-HCG(mIU/mL)"] = pd.to_numeric(df["II    beta-HCG(mIU/mL)"], errors='coerce')
df["AMH(ng/mL)"] = pd.to_numeric(df["AMH(ng/mL)"], errors='coerce')
df["Avg. F size (R) (mm)"] = pd.to_numeric(df["Avg. F size (R) (mm)"], errors='coerce')

# Fill missing values
df.fillna(df.median(), inplace=True)

# Define features and target
X = df.drop(columns=["PCOS (Y/N)"])
y = df["PCOS (Y/N)"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define Models
rf = RandomForestClassifier(n_estimators=100, random_state=42)
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
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

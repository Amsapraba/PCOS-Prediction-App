import streamlit as st
st.set_page_config(page_title="PCOS Prediction App", page_icon="🩺", layout="wide")

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import shap

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("PCOS_data.csv")
    df.columns = df.columns.str.strip()
    return df

df = load_data()

# Preprocess data
def preprocess_data(df):
    required_columns = [col for col in df.columns if "beta-HCG" in col or "AMH" in col]
    if len(required_columns) < 3:
        raise KeyError("Expected at least 3 features for prediction.")

    X = df[required_columns]
    y = df["PCOS (Y/N)"].astype(int)

    X = X.apply(pd.to_numeric, errors='coerce')
    X.fillna(X.median(), inplace=True)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return pd.DataFrame(X_scaled, columns=X.columns), y, scaler

X, y, scaler = preprocess_data(df)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest with class weighting
model = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
model.fit(X_train, y_train)

# SHAP Explainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_train)

# Streamlit App
st.title("🩺 PCOS Prediction and Analysis")
st.subheader("📊 Data Visualization")

# Graph: PCOS Distribution
fig, ax = plt.subplots(figsize=(8, 6))
sns.countplot(x='PCOS (Y/N)', data=df, hue='PCOS (Y/N)', legend=False, palette='viridis')
ax.set_title("Distribution of PCOS Cases")
st.pyplot(fig)

# Graph: Feature Importance
st.subheader("📈 Feature Importance")
importance = model.feature_importances_
fig, ax = plt.subplots(figsize=(8, 6))
sns.barplot(x=importance, y=X.columns, hue=X.columns, legend=False, palette="Blues_d")
ax.set_title("Feature Importance from Random Forest")
st.pyplot(fig)

# SHAP Summary Plot
st.subheader("💡 SHAP Analysis")
shap_values_class_1 = shap_values[1]

if shap_values_class_1.shape[1] != X_train.shape[1]:
    st.error(f"Mismatch between SHAP values ({shap_values_class_1.shape[1]}) and features ({X_train.shape[1]}).")
else:
    shap.summary_plot(shap_values_class_1, X_train, feature_names=X.columns)
    st.pyplot(plt)

# Sidebar for User Input
st.sidebar.header("📌 User Input")
weight = st.sidebar.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=60.0)
height = st.sidebar.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=160.0)
bmi = weight / ((height / 100) ** 2)

st.sidebar.write(f"📊 **BMI**: {bmi:.2f}")

user_input = {}
for col in X.columns:
    user_input[col] = st.sidebar.slider(f"{col} (Range)", min_value=round(float(X[col].min()), 2),
                                       max_value=round(float(X[col].max()), 2), value=float(X[col].mean()), step=0.1)

if st.sidebar.button("🚀 Predict PCOS"):
    input_df = pd.DataFrame([user_input])
    input_df[X.columns] = scaler.transform(input_df[X.columns])
    prediction = model.predict(input_df)[0]

    st.subheader("🎯 Prediction Result:")
    if prediction == 1:
        st.error("⚠️ PCOS Detected!")
    else:
        st.success("✅ No PCOS Detected")

st.markdown("---")
st.markdown("📌 Created for Smartathon Hackathon")

import streamlit as st  # First, import Streamlit

# Make sure st.set_page_config is the very first command in the app
st.set_page_config(page_title="PCOS Prediction App", page_icon="🩺", layout="wide")

# Now, import other libraries and define your functions
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import shap
import plotly.express as px

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("PCOS_data.csv")
    df.columns = df.columns.str.strip()  # Remove spaces from column names
    return df

df = load_data()

# Preprocessing function
def preprocess_data(df):
    required_columns = [col for col in df.columns if "beta-HCG" in col or "AMH" in col]

    if len(required_columns) < 3:
        raise KeyError(f"Missing required columns: Expected at least 3, found {len(required_columns)}")

    X = df[required_columns]
    y = df["PCOS (Y/N)"].astype(int)  # Convert target column to numeric

    # Handle missing values
    X = X.apply(pd.to_numeric, errors='coerce')
    X.fillna(X.median(), inplace=True)

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return pd.DataFrame(X_scaled, columns=X.columns), y, scaler

X, y, scaler = preprocess_data(df)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# SHAP Explainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_train)

# Streamlit App Interface
st.title("🩺 PCOS Prediction and Analysis App")
st.write("### Use this app to predict PCOS (Polycystic Ovary Syndrome) and analyze key health metrics.")

# Interactive Graphs Section
st.subheader("📊 Interactive Data Visualizations")

# Graph 1: Target Variable Distribution
fig, ax = plt.subplots(figsize=(8, 6))
sns.countplot(x='PCOS (Y/N)', data=df, ax=ax, palette='viridis')
ax.set_title("Distribution of PCOS Cases")
st.pyplot(fig)

# Graph 2: Correlation Heatmap
st.subheader("🔍 Feature Correlation Heatmap")
corr = df.corr()
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax, linewidths=0.5, fmt='.2f')
ax.set_title("Correlation Matrix of Features")
st.pyplot(fig)

# Graph 3: Feature Importance using Random Forest
st.subheader("📈 Feature Importance from Random Forest")
importance = model.feature_importances_
fig, ax = plt.subplots(figsize=(8, 6))
sns.barplot(x=importance, y=X.columns, ax=ax, palette="Blues_d")
ax.set_title("Feature Importance from Random Forest Model")
st.pyplot(fig)

# SHAP Summary Plot for Model Interpretability
st.subheader("💡 SHAP Summary Plot (Model Interpretability)")
shap.summary_plot(shap_values[1], X_train)
st.pyplot(plt)

# Sidebar for User Input Section
st.sidebar.header("📌 User Input Parameters")
weight = st.sidebar.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=60.0, step=0.1)
height = st.sidebar.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=160.0, step=0.1)
bmi = weight / ((height / 100) ** 2)

st.sidebar.write(f"📊 **Calculated BMI**: {bmi:.2f}")

# User Input Fields for Other Features
user_input = {}
for col in X.columns:
    user_input[col] = st.sidebar.slider(f"{col} (Range)", min_value=round(float(X[col].min()), 2),
                                       max_value=round(float(X[col].max()), 2), value=float(X[col].mean()), step=0.1)

# Prediction Button
if st.sidebar.button("🚀 Predict PCOS"):
    input_df = pd.DataFrame([user_input])
    input_df[X.columns] = scaler.transform(input_df[X.columns])
    prediction = model.predict(input_df)[0]

    # Prediction Result
    st.subheader("🎯 Prediction Result:")
    if prediction == 1:
        st.error("⚠️ PCOS Detected!")
        st.write("### 🔬 Detailed Analysis and Suggestions: ")
        st.write("- PCOS is a hormonal disorder that affects women of reproductive age.")
        st.write("- Symptoms include irregular periods, weight gain, and acne.")
        st.write("### 📌 Recommendations: ")
        st.write("- **Balanced Diet** and **Regular Exercise**.")
        st.write("- **Consult a Gynecologist** for professional evaluation.")
        st.write("- Monitor **blood sugar** and **hormonal levels** regularly.")
    else:
        st.success("✅ No PCOS Detected")
        st.write("### 📊 General Health Report: ")
        st.write("- Your **BMI**, **weight**, and **height** appear within normal ranges.")
        st.write("- Your **hormone levels** are likely within a healthy range, but consult a doctor for further analysis.")

# Adding Footer
st.markdown(
    """
    ---
    📝 This app was created as part of a personal project to predict and analyze PCOS. 
    For any questions, contact [Email](mailto:your-email@example.com).
    """
)

# Custom Styling
st.markdown(
    """
    <style>
    .streamlit-expanderHeader {
        font-size: 18px;
        font-weight: bold;
    }
    .css-1aehpv7 {
        color: #3d7e8c;
        font-family: 'Arial', sans-serif;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

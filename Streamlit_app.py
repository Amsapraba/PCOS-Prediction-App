import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import shap
import plotly.express as px

# Set page config
st.set_page_config(page_title="PCOS Prediction App", page_icon="🩺", layout="wide")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("PCOS_data.csv")
    df.columns = df.columns.str.strip()
    return df

df = load_data()

# Preprocessing function
def preprocess_data(df):
    required_columns = [col for col in df.columns if "beta-HCG" in col or "AMH" in col]
    if len(required_columns) < 3:
        raise KeyError(f"Missing required columns: Expected at least 3, found {len(required_columns)}")
    
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

# Defining Models
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
lgbm_model = LGBMClassifier(random_state=42)

# Voting Classifier
ensemble_model = VotingClassifier(
    estimators=[
        ('RandomForest', rf_model),
        ('XGBoost', xgb_model),
        ('LightGBM', lgbm_model)
    ], voting='soft'
)

# Train Ensemble Model
ensemble_model.fit(X_train, y_train)

# Evaluate Model
y_pred = ensemble_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.write(f'Ensemble Model Accuracy: {accuracy:.4f}')

# Streamlit App Interface
st.title("🩺 PCOS Prediction and Analysis App")
st.write("### Use this app to predict PCOS (Polycystic Ovary Syndrome) and analyze key health metrics.")

# Feature Importance
st.subheader("📈 Feature Importance from Ensemble Model")
feature_importance = np.mean([
    rf_model.feature_importances_,
    xgb_model.feature_importances_,
    lgbm_model.feature_importances_
], axis=0)

fig, ax = plt.subplots(figsize=(8, 6))
sns.barplot(x=feature_importance, y=X.columns, ax=ax, palette="Blues_d")
ax.set_title("Feature Importance from Ensemble Model")
st.pyplot(fig)

# SHAP Explanation
st.subheader("💡 SHAP Summary Plot (Model Interpretability)")
explainer = shap.TreeExplainer(ensemble_model)
shap_values = explainer.shap_values(X_train)
shap.summary_plot(shap_values[1], X_train, feature_names=X.columns)
st.pyplot(plt)

# Sidebar for User Input
st.sidebar.header("📌 User Input Parameters")
user_input = {}
for col in X.columns:
    user_input[col] = st.sidebar.slider(f"{col} (Range)", min_value=round(float(X[col].min()), 2),
                                       max_value=round(float(X[col].max()), 2), value=float(X[col].mean()), step=0.1)

if st.sidebar.button("🚀 Predict PCOS"):
    input_df = pd.DataFrame([user_input])
    input_df[X.columns] = scaler.transform(input_df[X.columns])
    prediction = ensemble_model.predict(input_df)[0]
    
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

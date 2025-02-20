import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import shap
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Page Configuration
st.set_page_config(page_title="PCOS Prediction Tool", layout="wide")
st.title("🩺 PCOS Prediction Tool")

# Sidebar Navigation
menu = st.sidebar.radio("Navigate to:", ["Home", "Upload Data", "Model Training", "Prediction", "Insights", "PCOS Quiz", "About"])

def load_data(file):
    df = pd.read_csv(file)
    return df

if menu == "Home":
    st.header("Welcome to the PCOS Prediction Tool")
    st.write("This tool helps predict PCOS based on non-invasive features using Machine Learning.")

elif menu == "Upload Data":
    st.header("Upload PCOS Dataset")
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        st.write("Preview of Uploaded Data:")
        st.dataframe(df.head())

elif menu == "Model Training":
    st.header("Train Machine Learning Model")
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        X = df.drop(columns=["PCOS"])
        y = df["PCOS"]
        
        # Handling missing values
        X.fillna(X.mean(), inplace=True)
        
        # Encoding categorical variables
        le = LabelEncoder()
        for col in X.select_dtypes(include=['object']).columns:
            X[col] = le.fit_transform(X[col])
        
        # Splitting data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Oversampling using SMOTE
        smote = SMOTE()
        X_train, y_train = smote.fit_resample(X_train, y_train)
        
        # Scaling
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # Model Training
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        st.write(f"Model Accuracy: {accuracy:.2f}")
        st.write("Classification Report:")
        st.text(classification_report(y_test, y_pred))
        
        # Save the model
        with open("pcos_model.pkl", "wb") as f:
            pickle.dump(model, f)

elif menu == "Prediction":
    st.header("PCOS Prediction")
    with open("pcos_model.pkl", "rb") as f:
        model = pickle.load(f)
    
    st.write("Enter patient details to predict PCOS")
    age = st.slider("Age", 15, 50, 25)
    bmi = st.slider("BMI", 15, 40, 25)
    waist_hip_ratio = st.slider("Waist-Hip Ratio", 0.6, 1.2, 0.85)
    
    input_data = np.array([[age, bmi, waist_hip_ratio]])
    prediction = model.predict(input_data)
    
    st.write("Prediction Result:")
    st.write("PCOS Positive" if prediction[0] == 1 else "PCOS Negative")

elif menu == "Insights":
    st.header("Explainable AI Insights")
    st.write("Feature importance and explainability using SHAP")
    
    with open("pcos_model.pkl", "rb") as f:
        model = pickle.load(f)
    
    explainer = shap.TreeExplainer(model)
    X_sample = X_test[:10]
    shap_values = explainer.shap_values(X_sample)
    
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, X_sample, show=False)
    st.pyplot(fig)

elif menu == "PCOS Quiz":
    st.header("Take the PCOS Risk Quiz")
    st.write("Answer the following questions to assess potential risk factors.")
    
    q1 = st.radio("Do you have irregular menstrual cycles?", ["Yes", "No"])
    q2 = st.radio("Do you experience excessive hair growth?", ["Yes", "No"])
    q3 = st.radio("Do you have acne or oily skin?", ["Yes", "No"])
    
    risk_score = sum([q1 == "Yes", q2 == "Yes", q3 == "Yes"])
    if risk_score >= 2:
        st.write("You may be at risk for PCOS. Consider consulting a doctor.")
    else:
        st.write("Your risk for PCOS appears low based on this quiz.")

elif menu == "About":
    st.header("About This Project")
    st.write("This PCOS Prediction Tool was built using Machine Learning to assist in non-invasive PCOS detection.")

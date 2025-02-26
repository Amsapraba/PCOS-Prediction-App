import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import shap
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from lime.lime_tabular import LimeTabularExplainer

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
        
        # Model Training with Ensemble Learning
        rf = RandomForestClassifier()
        xgb = XGBClassifier()
        lgbm = LGBMClassifier()
        
        rf.fit(X_train, y_train)
        xgb.fit(X_train, y_train)
        lgbm.fit(X_train, y_train)
        
        y_pred_rf = rf.predict(X_test)
        y_pred_xgb = xgb.predict(X_test)
        y_pred_lgbm = lgbm.predict(X_test)
        
        accuracy_rf = accuracy_score(y_test, y_pred_rf)
        accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
        accuracy_lgbm = accuracy_score(y_test, y_pred_lgbm)
        
        st.write(f"Random Forest Accuracy: {accuracy_rf:.2f}")
        st.write(f"XGBoost Accuracy: {accuracy_xgb:.2f}")
        st.write(f"LightGBM Accuracy: {accuracy_lgbm:.2f}")
        
        # Save the best model (based on accuracy)
        best_model = max([(rf, accuracy_rf), (xgb, accuracy_xgb), (lgbm, accuracy_lgbm)], key=lambda x: x[1])[0]
        with open("pcos_model.pkl", "wb") as f:
            pickle.dump(best_model, f)

elif menu == "Prediction":
    st.header("PCOS Prediction")
    if os.path.exists("pcos_model.pkl"):
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
    else:
        st.error("Model file not found. Please train the model first in the 'Model Training' section.")

elif menu == "PCOS Quiz":
    st.header("Take the PCOS Risk Quiz")
    st.write("Answer the following questions to assess potential risk factors.")
    
    questions = [
        "Do you have irregular menstrual cycles?",
        "Do you experience excessive hair growth?",
        "Do you have acne or oily skin?",
        "Do you frequently feel fatigued or have mood swings?",
        "Do you experience sudden weight gain or difficulty losing weight?",
        "Do you have dark skin patches (Acanthosis Nigricans)?",
        "Does PCOS run in your family?"
    ]
    
    responses = [st.radio(q, ["Yes", "No"]) for q in questions]
    risk_score = sum([resp == "Yes" for resp in responses])
    
    if risk_score >= 4:
        st.write("You may be at high risk for PCOS. Consider consulting a doctor and maintaining a healthy lifestyle.")
    elif 2 <= risk_score < 4:
        st.write("You have moderate risk for PCOS. Focus on lifestyle management, exercise, and a balanced diet.")
    else:
        st.write("Your risk for PCOS appears low based on this quiz. Keep maintaining a healthy lifestyle!")

elif menu == "About":
    st.header("About This Project")
    st.write("This PCOS Prediction Tool was built using Machine Learning to assist in non-invasive PCOS detection.")

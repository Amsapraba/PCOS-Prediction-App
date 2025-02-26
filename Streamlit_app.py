import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Title
st.title("PCOS Prediction App")

# Upload dataset
uploaded_file = st.file_uploader("Upload your PCOS dataset (CSV)", type=["csv"])

if uploaded_file:
    # Load Data
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()  # Remove extra spaces in column names
    
    # Ensure the correct PCOS column exists
    pcos_column = None
    for col in df.columns:
        if "PCOS" in col:
            pcos_column = col
            break
    
    if pcos_column is None:
        st.error("Error: No 'PCOS' column found in the dataset. Please check your file.")
    else:
        # Convert categorical columns to numerical
        label_encoders = {}
        for col in df.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le
        
        # Fill missing values with column mean
        df.fillna(df.mean(), inplace=True)
        
        # Prepare data
        X = df.drop(columns=[pcos_column])  # Features
        y = df[pcos_column]  # Target
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Show model accuracy
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"### Model Accuracy: {accuracy:.2f}")
        
        # Display feature importance graph
        feature_importances = model.feature_importances_
        features = X.columns
        plt.figure(figsize=(10, 5))
        plt.barh(features, feature_importances, color='skyblue')
        plt.xlabel("Importance")
        plt.ylabel("Features")
        plt.title("Feature Importance in PCOS Prediction")
        st.pyplot(plt)
        
        # Prediction Section
        st.write("### Enter Details for Prediction")
        user_input = {}
        for col in X.columns:
            user_input[col] = st.number_input(f"{col}", value=float(df[col].mean()))
        
        if st.button("Predict"):
            input_df = pd.DataFrame([user_input])
            prediction = model.predict(input_df)[0]
            result = "PCOS Detected" if prediction == 1 else "No PCOS Detected"
            st.write(f"### Prediction: {result}")
        
        # Quiz Section
        st.write("### PCOS Risk Assessment Quiz")
        
        q1 = st.selectbox("Do you experience irregular periods?", ["No", "Sometimes", "Yes"])
        q2 = st.selectbox("Do you have excessive hair growth (hirsutism)?", ["No", "Mild", "Severe"])
        q3 = st.selectbox("Do you experience frequent acne or oily skin?", ["No", "Sometimes", "Yes"])
        q4 = st.selectbox("Have you noticed unexplained weight gain?", ["No", "Mild", "Significant"])
        q5 = st.selectbox("Do you frequently consume processed or high-sugar foods?", ["No", "Sometimes", "Yes"])
        
        if st.button("Check PCOS Proneness"):
            score = sum([q1 == "Yes", q2 == "Severe", q3 == "Yes", q4 == "Significant", q5 == "Yes"])
            if score >= 3:
                st.write("### High Risk: You might be prone to PCOS. Consider consulting a doctor.")
            elif score == 2:
                st.write("### Moderate Risk: You may have some risk factors. Keep monitoring your health.")
            else:
                st.write("### Low Risk: You have minimal risk of PCOS.")

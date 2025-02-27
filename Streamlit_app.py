import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Title
st.title("PCOS Prediction App")

# Upload dataset
uploaded_file = st.file_uploader("Upload your PCOS dataset (CSV)", type=["csv"])

if uploaded_file:
    # Load Data
    df = pd.read_csv(uploaded_file)
    
    # Ensure 'PCOS' column exists
    if "PCOS" not in df.columns:
        st.error("Error: 'PCOS' column not found. Please check your dataset.")
    else:
        # Show sample data
        st.write("### Sample Data", df.head())

        # Prepare data
        X = df.drop(columns=["PCOS"])  # Features
        y = df["PCOS"]  # Target

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Show model accuracy
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"### Model Accuracy: {accuracy:.2f}")

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

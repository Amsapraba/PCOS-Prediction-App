import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
def load_data():
    df = pd.read_csv("PCOS_data.csv")  # Replace with actual dataset path
    return df

def preprocess_data(df):
    # Assuming target column is 'PCOS' and other necessary preprocessing steps
    X = df.drop(columns=['PCOS'])
    y = df['PCOS']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return accuracy, report

# Streamlit UI
st.title("PCOS Prediction App")

st.sidebar.header("Upload PCOS Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Sample Data")
    st.write(df.head())
    
    X_train, X_test, y_train, y_test = preprocess_data(df)
    model = train_model(X_train, y_train)
    accuracy, report = evaluate_model(model, X_test, y_test)
    
    st.write(f"### Model Accuracy: {accuracy:.2f}")
    st.text(report)

    # Display feature importance graph
    feature_importances = model.feature_importances_
    features = X_train.columns
    plt.figure(figsize=(10, 5))
    plt.barh(features, feature_importances, color='skyblue')
    plt.xlabel("Importance")
    plt.ylabel("Features")
    plt.title("Feature Importance in PCOS Prediction")
    st.pyplot(plt)

    # Prediction Section
    st.write("### Enter Details for Prediction")
    user_input = {}
    for col in X_train.columns:
        user_input[col] = st.number_input(f"{col}", value=float(df[col].mean()))
    
    if st.button("Predict"):
        input_df = pd.DataFrame([user_input])
        prediction = model.predict(input_df)
        result = "PCOS Detected" if prediction[0] == 1 else "No PCOS Detected"
        st.write(f"### Prediction: {result}")

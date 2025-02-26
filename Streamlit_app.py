import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the trained model
@st.cache_resource
def load_model():
    return joblib.load("pcos_model.pkl")

model = load_model()

# Streamlit UI
st.set_page_config(page_title="PCOS Prediction App", layout="centered")
st.title("🌸 PCOS Prediction App")

# User input fields
age = st.number_input("Enter Age", min_value=10, max_value=60, value=25)
weight = st.number_input("Enter Weight (kg)", min_value=30, max_value=200, value=65)
height = st.number_input("Enter Height (cm)", min_value=100, max_value=200, value=160)

# Convert to BMI
bmi = weight / ((height / 100) ** 2)

# Predict button
if st.button("🔍 Predict PCOS"):
    input_data = pd.DataFrame([[age, bmi]], columns=["Age", "BMI"])  # Adjust based on dataset
    prediction = model.predict(input_data)
    
    if prediction[0] == 1:
        st.error("⚠️ High risk of PCOS detected! Please consult a doctor.")
    else:
        st.success("✅ You are healthy! No signs of PCOS detected.")

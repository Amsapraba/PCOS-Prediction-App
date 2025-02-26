import streamlit as st
import joblib
import numpy as np

# Load the trained model and scaler
@st.cache_resource
def load_model():
    return joblib.load("pcos_model.pkl")

@st.cache_resource
def load_scaler():
    return joblib.load("scaler.pkl")

model = load_model()
scaler = load_scaler()

# Streamlit UI
st.title("PCOS Prediction App")

# User input fields
age = st.number_input("Enter Age", min_value=10, max_value=60, value=25)
weight = st.number_input("Enter Weight (kg)", min_value=30, max_value=200, value=65)
height = st.number_input("Enter Height (cm)", min_value=100, max_value=200, value=160)
cycle_length = st.number_input("Enter Cycle Length (days)", min_value=15, max_value=50, value=28)
follicle_count = st.number_input("Enter Follicle Count (Left Ovary)", min_value=0, max_value=30, value=5)

# Convert height to meters and calculate BMI
bmi = weight / ((height / 100) ** 2)

# Prepare input data
input_data = np.array([[age, bmi, cycle_length, follicle_count]])
scaled_data = scaler.transform(input_data)

# Predict button
if st.button("Predict PCOS"):
    prediction = model.predict(scaled_data)
    
    if prediction[0] == 1:
        st.error("⚠️ High risk of PCOS detected! Please consult a doctor.")
    else:
        st.success("✅ You are healthy! No signs of PCOS detected.")

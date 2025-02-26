import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
import base64
from fpdf import FPDF

# Load trained model
model = joblib.load("pcos_model.pkl")

# Function to generate PDF report
def generate_pdf(prediction, recommendations):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", style='', size=14)
    pdf.cell(200, 10, "PCOS Prediction Report", ln=True, align='C')
    pdf.ln(10)
    pdf.cell(200, 10, f"Prediction: {prediction}", ln=True)
    pdf.ln(10)
    pdf.multi_cell(0, 10, f"Recommendations: {recommendations}")
    pdf_filename = "pcos_report.pdf"
    pdf.output(pdf_filename)
    return pdf_filename

# Streamlit UI
st.set_page_config(page_title="PCOS Prediction", layout="wide")
st.title("🌸 PCOS Prediction App")

# Sidebar Virtual Assistant
def virtual_assistant():
    st.sidebar.title("👩‍⚕️ Virtual Assistant")
    st.sidebar.write("Hey, how may I assist you?")

virtual_assistant()

# Dashboard Navigation
menu = ["Home", "Predict PCOS", "Health Games", "Quiz", "Generate Report"]
choice = st.sidebar.radio("Navigation", menu)

if choice == "Home":
    st.header("Welcome to the PCOS Prediction App!")
    st.write("This AI-powered tool helps in early detection of PCOS using health parameters.")

elif choice == "Predict PCOS":
    st.header("🔍 PCOS Prediction")
    
    # User Inputs
    age = st.number_input("Age", min_value=15, max_value=50, value=25)
    bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=22.0)
    cycle_length = st.number_input("Cycle Length (days)", min_value=15, max_value=60, value=28)
    irregular_periods = st.radio("Irregular Periods?", ["Yes", "No"])
    weight_gain = st.radio("Sudden Weight Gain?", ["Yes", "No"])
    hair_growth = st.radio("Excessive Hair Growth?", ["Yes", "No"])
    acne = st.radio("Frequent Acne Issues?", ["Yes", "No"])
    
    # Convert categorical to numerical
    inputs = [age, bmi, cycle_length, int(irregular_periods == "Yes"), int(weight_gain == "Yes"), int(hair_growth == "Yes"), int(acne == "Yes")]
    
    if st.button("Predict Now"):
        prediction = model.predict([inputs])
        if prediction[0] == 1:
            st.error("⚠️ High risk of PCOS detected. Consult a doctor.")
            recommendations = "Maintain a balanced diet, exercise regularly, and consult a gynecologist."
        else:
            st.success("✅ You are healthy! Keep maintaining a good lifestyle.")
            recommendations = "Continue with a healthy diet and regular check-ups."
        
        # Generate PDF Report
        pdf_file = generate_pdf("High Risk" if prediction[0] == 1 else "Healthy", recommendations)
        with open(pdf_file, "rb") as f:
            pdf_data = f.read()
        b64 = base64.b64encode(pdf_data).decode()
        href = f'<a href="data:application/octet-stream;base64,{b64}" download="{pdf_file}">📄 Download Report</a>'
        st.markdown(href, unsafe_allow_html=True)

elif choice == "Health Games":
    st.header("🎮 Fun Health Games")
    st.write("Coming Soon! Stay tuned for interactive games to improve health awareness.")

elif choice == "Quiz":
    st.header("🧠 Health Awareness Quiz")
    question = "What lifestyle change helps in managing PCOS?"
    options = ["A. Eating junk food", "B. Regular exercise", "C. Skipping meals"]
    answer = st.radio(question, options)
    if st.button("Submit Answer"):
        if answer == "B. Regular exercise":
            st.success("Correct! Regular exercise helps in PCOS management.")
        else:
            st.error("Incorrect! Try again.")

elif choice == "Generate Report":
    st.header("📄 Generate Personalized Report")
    st.write("Click below to generate your instant PDF health report.")
    if st.button("Generate Report Now"):
        pdf_file = generate_pdf("Healthy", "Maintain a good lifestyle")
        with open(pdf_file, "rb") as f:
            pdf_data = f.read()
        b64 = base64.b64encode(pdf_data).decode()
        href = f'<a href="data:application/octet-stream;base64,{b64}" download="{pdf_file}">📄 Download Report</a>'
        st.markdown(href, unsafe_allow_html=True)

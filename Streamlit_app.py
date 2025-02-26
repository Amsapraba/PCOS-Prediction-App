import joblib
import pandas as pd
import os
import streamlit as st

# Streamlit UI setup
st.set_page_config(page_title="PCOS Prediction", layout="wide")
st.title("🌸 PCOS Prediction & Health Analysis")

# Load the trained model
model_path = "pcos_model.pkl"
if os.path.exists(model_path):
    model = joblib.load(model_path)
    st.success("✅ Model loaded successfully!")
else:
    st.error("🚨 Model file not found! Please upload 'pcos_model.pkl' to the repository.")

# Load the dataset for visualization (if needed)
csv_path = "pcos_data.csv"
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    st.success("✅ Dataset loaded successfully!")
else:
    st.warning("⚠️ CSV file not found! Some visualizations may not work.")

# Display dataset preview
if "df" in locals():
    st.write("### Sample Data:")
    st.dataframe(df.head())

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Predict PCOS", "Data Visualization", "Health Quiz"])

# Home Page
if page == "Home":
    st.subheader("Welcome to the PCOS Prediction App! 🚀")
    st.write("This app predicts PCOS using an AI model and provides health insights.")

# Prediction Page
elif page == "Predict PCOS":
    st.subheader("🔍 Predict PCOS")
    st.write("Enter your details to check for PCOS risk.")

    # Example input fields (Modify according to your dataset)
    age = st.number_input("Age", min_value=10, max_value=50, value=25)
    bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=22.5)
    cycle = st.number_input("Cycle Length (days)", min_value=10, max_value=50, value=28)

    if st.button("Predict"):
        if "model" in locals():
            input_data = [[age, bmi, cycle]]
            prediction = model.predict(input_data)
            if prediction[0] == 1:
                st.error("⚠️ High risk of PCOS detected! Consult a doctor.")
            else:
                st.success("✅ No PCOS risk detected! Maintain a healthy lifestyle.")
        else:
            st.error("Model not loaded. Please check!")

# Data Visualization Page
elif page == "Data Visualization":
    st.subheader("📊 Data Insights")
    if "df" in locals():
        st.write("Data statistics:")
        st.write(df.describe())
    else:
        st.warning("No dataset found!")

# Health Quiz Page
elif page == "Health Quiz":
    st.subheader("🧠 PCOS Awareness Quiz")
    st.write("Test your knowledge about PCOS!")

    question1 = st.radio("Q1: What is a common symptom of PCOS?", ["Weight loss", "Irregular periods", "High energy"])
    if st.button("Submit Quiz"):
        if question1 == "Irregular periods":
            st.success("✅ Correct!")
        else:
            st.error("❌ Incorrect! A common symptom is irregular periods.")

# Footer
st.markdown("---")
st.write("Developed with ❤️ for Smartathon Hackathon")


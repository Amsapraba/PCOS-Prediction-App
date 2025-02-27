import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Load the trained model and scaler
@st.cache_resource
def load_model():
    model = joblib.load("/content/pcos_random_forest_model.pkl")  # Path to saved model
    scaler = joblib.load("/content/scaler.pkl")  # Path to saved scaler
    return model, scaler

model, scaler = load_model()

# Define the expected feature names (must match training data)
feature_names = [
    'Age (yrs)', 'Weight (Kg)', 'Height(Cm)', 'BMI', 'Pulse rate(bpm)', 'Hb(g/dl)',
    'Cycle(R/I)', 'Cycle length(days)', 'Marraige Status (Yrs)', 'Pregnant(Y/N)',
    'No. of aborptions', 'Blood Group', 'Fast food (Y/N)', 'Reg.Exercise(Y/N)',
    'I   beta-HCG(mIU/mL)', 'II    beta-HCG(mIU/mL)', 'FSH(mIU/mL)', 'LH(mIU/mL)',
    'FSH/LH', 'Hip(inch)', 'Waist(inch)', 'TSH (mIU/L)', 'AMH(ng/mL)', 'PRL(ng/mL)',
    'Vit D3 (ng/mL)', 'PRG(ng/mL)', 'RBS(mg/dl)', 'Weight gain(Y/N)',
    'hair growth(Y/N)', 'Skin darkening (Y/N)', 'Hair loss(Y/N)', 'Pimples(Y/N)',
    'Follicle No. (L)', 'Follicle No. (R)', 'Avg. F size (L) (mm)',
    'Avg. F size (R) (mm)', 'Endometrium (mm)', 'BP _Systolic (mmHg)',
    'BP _Diastolic (mmHg)', 'Waist:Hip Ratio', 'LH/FSH Ratio'
]

# Streamlit UI
st.title("PCOS Prediction App")
st.write("Enter the required values to check if PCOS is detected or not.")

# User input fields
user_input = []
for i, feature in enumerate(feature_names):
    value = st.number_input(f"Enter {feature}", min_value=0.0, step=0.1, key=f"{feature}_{i}")
    user_input.append(value)

# Prediction Button
if st.button("Predict PCOS"):
    # Convert input to numpy array and reshape
    input_array = np.array([user_input])

    # Debugging: Check the number of features
    if input_array.shape[1] != len(feature_names):
        st.error(f"Expected {len(feature_names)} features, but got {input_array.shape[1]} features.")
    else:
        # Scale the input and predict
        input_scaled = scaler.transform(input_array)
        prediction = model.predict(input_scaled)

        # Output result
        result = "PCOS Detected" if prediction[0] == 1 else "No PCOS Detected"
        st.subheader(f"Prediction Result: {result}")import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Load the trained model and scaler
@st.cache_resource
def load_model():
    model = joblib.load("/content/pcos_random_forest_model.pkl")  # Path to saved model
    scaler = joblib.load("/content/scaler.pkl")  # Path to saved scaler
    return model, scaler

model, scaler = load_model()

# Define the expected feature names (must match training data)
feature_names = [
    'Age (yrs)', 'Weight (Kg)', 'Height(Cm)', 'BMI', 'Pulse rate(bpm)', 'Hb(g/dl)',
    'Cycle(R/I)', 'Cycle length(days)', 'Marraige Status (Yrs)', 'Pregnant(Y/N)',
    'No. of aborptions', 'Blood Group', 'Fast food (Y/N)', 'Reg.Exercise(Y/N)',
    'I   beta-HCG(mIU/mL)', 'II    beta-HCG(mIU/mL)', 'FSH(mIU/mL)', 'LH(mIU/mL)',
    'FSH/LH', 'Hip(inch)', 'Waist(inch)', 'TSH (mIU/L)', 'AMH(ng/mL)', 'PRL(ng/mL)',
    'Vit D3 (ng/mL)', 'PRG(ng/mL)', 'RBS(mg/dl)', 'Weight gain(Y/N)',
    'hair growth(Y/N)', 'Skin darkening (Y/N)', 'Hair loss(Y/N)', 'Pimples(Y/N)',
    'Follicle No. (L)', 'Follicle No. (R)', 'Avg. F size (L) (mm)',
    'Avg. F size (R) (mm)', 'Endometrium (mm)', 'BP _Systolic (mmHg)',
    'BP _Diastolic (mmHg)', 'Waist:Hip Ratio', 'LH/FSH Ratio'
]

# Streamlit UI
st.title("PCOS Prediction App")
st.write("Enter the required values to check if PCOS is detected or not.")

# User input fields
user_input = []
for i, feature in enumerate(feature_names):
    value = st.number_input(f"Enter {feature}", min_value=0.0, step=0.1, key=f"{feature}_{i}")
    user_input.append(value)

# Prediction Button
if st.button("Predict PCOS"):
    # Convert input to numpy array and reshape
    input_array = np.array([user_input])

    # Debugging: Check the number of features
    if input_array.shape[1] != len(feature_names):
        st.error(f"Expected {len(feature_names)} features, but got {input_array.shape[1]} features.")
    else:
        # Scale the input and predict
        input_scaled = scaler.transform(input_array)
        prediction = model.predict(input_scaled)

        # Output result
        result = "PCOS Detected" if prediction[0] == 1 else "No PCOS Detected"
        st.subheader(f"Prediction Result: {result}")

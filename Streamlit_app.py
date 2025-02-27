import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


@st.cache_data  
def load_model():
    try:
        model_path = "pcos_random_forest_model.pkl"
        scaler_path = "scaler.pkl"
        
        if not os.path.exists(model_path):
            st.error("❌ Model file not found! Please check the file path.")
            return None, None
        if not os.path.exists(scaler_path):
            st.error("❌ Scaler file not found! Please check the file path.")
            return None, None
        
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    except Exception as e:
        st.error(f"⚠️ Error loading model: {e}")
        return None, None


model, scaler = load_model()


if model is None or scaler is None:
    st.stop()  


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


st.title("PCOS Prediction App")
st.write("Enter the required values to check if PCOS is detected or not.")


user_input = []
for i, feature in enumerate(feature_names):
    value = st.number_input(f"Enter {feature}", min_value=0.0, step=0.1, key=f"{feature}_{i}")
    user_input.append(value)


if st.button("Predict PCOS"):
   
    input_array = np.array([user_input])
    
    
    st.write(f"Input Shape: {input_array.shape}, Expected Features: {len(feature_names)}")
    if input_array.shape[1] != len(feature_names):
        st.error(f"🚨 Expected {len(feature_names)} features, but got {input_array.shape[1]}.")
    else:
        try:
            input_scaled = scaler.transform(input_array)
            prediction = model.predict(input_scaled)
            result = "PCOS Detected" if prediction[0] == 1 else "No PCOS Detected"
            st.subheader(f"Prediction Result: {result}")
        except Exception as e:
            st.error(f"⚠️ Prediction Error: {e}")

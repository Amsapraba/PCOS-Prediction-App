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

# Sidebar navigation
menu = st.sidebar.radio("Navigation", ["Home", "PCOS Prediction", "Quiz", "Health Recipes", "Personalized Meal Plan", "Games", "Mood Tracker"])

if menu == "Home":
    st.write("## Hey there! What’s up? Click on any of the features in the dashboard to get started.")
    st.write("Use this tool to predict PCOS risk, take a quiz to assess symptoms, and explore healthy recipes!")

elif menu == "PCOS Prediction":
    uploaded_file = st.file_uploader("Upload your PCOS dataset (CSV)", type=["csv"])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        df.columns = df.columns.str.strip()
        
        pcos_column = None
        for col in df.columns:
            if "PCOS" in col:
                pcos_column = col
                break
        
        if pcos_column is None:
            st.error("Error: No 'PCOS' column found in the dataset. Please check your file.")
        else:
            label_encoders = {}
            for col in df.select_dtypes(include=['object']).columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                label_encoders[col] = le
            
            df.fillna(df.mean(), inplace=True)
            
            X = df.drop(columns=[pcos_column])
            y = df[pcos_column]
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            st.write(f"### Model Accuracy: {accuracy:.2f}")
            
            feature_importances = model.feature_importances_
            features = X.columns
            plt.figure(figsize=(10, 5))
            plt.barh(features, feature_importances, color='skyblue')
            plt.xlabel("Importance")
            plt.ylabel("Features")
            plt.title("Feature Importance in PCOS Prediction")
            st.pyplot(plt)
            
            st.write("### Enter Details for Prediction")
            user_input = {}
            for col in X.columns:
                user_input[col] = st.number_input(f"{col}", value=float(df[col].mean()))
            
            if st.button("Predict"):
                input_df = pd.DataFrame([user_input])
                prediction = model.predict(input_df)[0]
                result = "PCOS Detected" if prediction == 1 else "No PCOS Detected"
                st.write(f"### Prediction: {result}")

elif menu == "Quiz":
    st.write("### PCOS Awareness Quiz")
    questions = {
        "What is a common symptom of PCOS?": ["Weight gain", "Hair loss", "Irregular periods", "All of the above"],
        "Which hormone is often elevated in PCOS?": ["Estrogen", "Progesterone", "Insulin", "Testosterone"],
        "Which lifestyle change can help manage PCOS?": ["Exercise", "Balanced diet", "Stress reduction", "All of the above"],
        "Does stress impact PCOS?": ["Yes", "No"],
        "Can PCOS be cured completely?": ["Yes", "No, but it can be managed"]
    }
    
    for question, options in questions.items():
        answer = st.radio(question, options)
        if answer == options[-1]:
            st.write("✅ Correct!")
        else:
            st.write("❌ Incorrect, try again!")

elif menu == "Personalized Meal Plan":
    st.write("### Personalized Meal Plan")
    meal_plans = {
        "Low Carb Plan": ["Breakfast: Scrambled eggs with spinach", "Lunch: Grilled chicken with quinoa", "Dinner: Baked salmon with steamed vegetables"],
        "Balanced Diet Plan": ["Breakfast: Oatmeal with nuts", "Lunch: Brown rice with tofu stir-fry", "Dinner: Lentil soup with whole-grain bread"],
        "Vegetarian Plan": ["Breakfast: Smoothie with banana and almond milk", "Lunch: Chickpea salad with olive oil dressing", "Dinner: Stuffed bell peppers"]
    }
    selected_plan = st.selectbox("Select your meal plan", list(meal_plans.keys()))
    st.write(f"### {selected_plan}")
    for meal in meal_plans[selected_plan]:
        st.write(meal)

elif menu == "Games":
    st.write("### Health Games")
    st.write("Play simple games to track your health!")
    game_choice = st.radio("Choose a game:", ["Step Counter Challenge", "Healthy Plate Builder", "Stress Relief Breathing"])
    
    if game_choice == "Step Counter Challenge":
        steps = st.number_input("Enter your daily steps", min_value=0, value=5000)
        if steps >= 10000:
            st.write("🏆 Great job! You're meeting the daily activity goal.")
        else:
            st.write("🚶 Keep moving! Aim for at least 10,000 steps per day.")
    
    elif game_choice == "Healthy Plate Builder":
        st.write("Build a balanced plate with proteins, veggies, and grains!")
        protein = st.selectbox("Choose a protein source", ["Chicken", "Tofu", "Lentils", "Eggs"])
        veggies = st.selectbox("Choose a vegetable", ["Spinach", "Broccoli", "Carrots", "Peppers"])
        grains = st.selectbox("Choose a grain", ["Brown rice", "Quinoa", "Whole wheat bread", "Oats"])
        st.write(f"Your healthy plate: {protein}, {veggies}, and {grains} 🍽")
    
    elif game_choice == "Stress Relief Breathing":
        st.write("Take a deep breath in... Hold for 4 seconds... Exhale slowly...")
        st.write("Repeat this for 3-5 minutes to relieve stress.")

elif menu == "Mood Tracker":
    st.write("### Mood Tracker")
    
    if "mood_log" not in st.session_state:
        st.session_state.mood_log = []
    
    mood = st.radio("How are you feeling today?", ["Happy", "Stressed", "Tired", "Motivated"])
    note = st.text_area("Write a short journal entry about your mood (optional):")
    
    if st.button("Log Mood"):
        st.session_state.mood_log.append((mood, note))
        st.success("Mood logged successfully!")

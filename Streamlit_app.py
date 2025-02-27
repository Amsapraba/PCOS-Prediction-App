import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
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

menu = st.sidebar.selectbox("Menu", ["Home", "PCOS Prediction", "Symptom Quiz", "Healthy Recipes", "Personalized Meal Plan", "Games", "Mood Tracker"])

if menu == "Home":
    st.write("Use this tool to predict PCOS risk, take a quiz to assess symptoms, and explore healthy recipes!")

elif menu == "PCOS Prediction":
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

elif menu == "Symptom Quiz":
    st.write("### PCOS Symptom Quiz")
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

elif menu == "Healthy Recipes":
    st.write("### Healthy Recipes for PCOS")
    recipes = {
        "Green Smoothie": "- 1 cup spinach\n- 1 banana\n- 1/2 cup Greek yogurt\n- 1 cup almond milk\n- Blend and enjoy!",
        "Oatmeal with Chia Seeds": "- 1/2 cup rolled oats\n- 1 tbsp chia seeds\n- 1 cup almond milk\n- 1 tsp honey\n- Mix and let it sit for 10 minutes before eating.",
        "Avocado Toast with Egg": "- 1 slice whole-grain bread\n- 1/2 avocado\n- 1 boiled egg\n- Salt and pepper to taste\n- Mash avocado on toast, add sliced egg, and season."
    }
    selected_recipe = st.selectbox("Click to know more about a recipe", list(recipes.keys()))
    st.write(f"### {selected_recipe}")
    st.write(recipes[selected_recipe])

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

elif menu == "Mood Tracker":
    st.write("### Mood Tracker")
    mood = st.selectbox("How are you feeling today?", ["Happy", "Sad", "Anxious", "Stressed", "Neutral"])
    note = st.text_area("Journal your thoughts")
    if "mood_log" not in st.session_state:
        st.session_state.mood_log = []
    if st.button("Log Mood"):
        st.session_state.mood_log.append((mood, note))
        st.success("Mood logged successfully!")

import streamlit as st
import pandas as pd
import numpy as np

st.title("PCOS Prediction and Health Tool")
menu = st.sidebar.selectbox("Choose a feature", ["Home", "PCOS Prediction", "Symptom Quiz", "Health Recipes", "Personalized Meal Plan", "Games", "Mood Tracker"])

if menu == "Home":
    st.write("Use this tool to predict PCOS risk, take a quiz to assess symptoms, and explore healthy recipes!")

elif menu == "PCOS Prediction":
    # Upload dataset
    uploaded_file = st.file_uploader("Upload your PCOS dataset (CSV)", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("### Dataset Preview")
        st.dataframe(df.head())

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

elif menu == "Health Recipes":
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

elif menu == "Games":
    st.write("### Health Games")
    game_choice = st.radio("Choose a game:", ["Step Counter Challenge", "Healthy Plate Builder", "Stress Relief Breathing"])
    
    if game_choice == "Step Counter Challenge":
        steps = st.number_input("Enter your daily steps", min_value=0, value=5000)
        if steps >= 10000:
            st.write("🏆 Great job! You're meeting the daily activity goal.")
        else:
            st.write("🚶 Keep moving! Aim for at least 10,000 steps per day.")
    
    elif game_choice == "Healthy Plate Builder":
        protein = st.selectbox("Choose a protein source", ["Chicken", "Tofu", "Lentils", "Eggs"])
        veggies = st.selectbox("Choose a vegetable", ["Spinach", "Broccoli", "Carrots", "Peppers"])
        grains = st.selectbox("Choose a grain", ["Brown rice", "Quinoa", "Whole wheat bread", "Oats"])
        st.write(f"Your healthy plate: {protein}, {veggies}, and {grains} 🍽")
    
    elif game_choice == "Stress Relief Breathing":
        st.write("Take a deep breath in... Hold for 4 seconds... Exhale slowly...")
        st.write("Repeat this for 3-5 minutes to relieve stress.")

elif menu == "Mood Tracker":
    st.write("### Mood Tracker")
    mood = st.selectbox("How are you feeling today?", ["Happy", "Sad", "Stressed", "Relaxed", "Neutral"])
    note = st.text_area("Write a short journal entry about your mood:")
    
    if "mood_log" not in st.session_state:
        st.session_state.mood_log = []
    
    if st.button("Log Mood"):
        st.session_state.mood_log.append((mood, note))
        st.success("Mood logged successfully!")
    
    if st.session_state.mood_log:
        st.write("### Your Mood Log")
        df_mood = pd.DataFrame(st.session_state.mood_log, columns=["Mood", "Journal"])
        st.dataframe(df_mood)
        
        st.write("### Mood Trend")
        mood_counts = df_mood["Mood"].value_counts()
        st.bar_chart(mood_counts)

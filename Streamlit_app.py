import streamlit as st
import pandas as pd
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import re

# Streamlit App Setup
st.set_page_config(page_title="PCOS Prediction", layout="wide")
st.title("PCOS Prediction Model")

# File uploader
uploaded_file = st.file_uploader("Upload your PCOS dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    try:
        # Load Dataset
        df = pd.read_csv(uploaded_file)
        st.success("File uploaded successfully!")

        # Show dataset preview
        st.subheader("Dataset Preview")
        st.dataframe(df)

        # Fix column names: Remove special characters and replace spaces
        df.columns = [re.sub(r'[^a-zA-Z0-9_]', '', col).replace(" ", "_") for col in df.columns]

        # Identify target variable
        target_column = "PCOS_YN"  # Change based on dataset

        if target_column not in df.columns:
            st.error(f"Target column '{target_column}' not found! Check column names.")
        else:
            # Separate features and target
            X = df.drop(columns=[target_column])
            y = df[target_column]

            # Handle missing values
            X.fillna(0, inplace=True)

            # Convert categorical columns to numerical
            for col in X.select_dtypes(include=['object']).columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col])

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Ensure column names are valid
            X_train.columns = X_train.columns.astype(str)
            X_test.columns = X_test.columns.astype(str)

            # LightGBM dataset
            train_data = lgb.Dataset(X_train, label=y_train)
            valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

            # LightGBM parameters
            params = {
                "objective": "binary",
                "metric": "accuracy",
                "boosting_type": "gbdt",
                "learning_rate": 0.05,
                "num_leaves": 31,
                "verbose": -1
            }

            # Train model
            model = lgb.train(params, train_data, valid_sets=[valid_data], verbose_eval=10)

            # Predictions
            y_pred = model.predict(X_test)
            y_pred_binary = np.round(y_pred)

            # Accuracy
            accuracy = accuracy_score(y_test, y_pred_binary)
            st.subheader("Model Performance")
            st.write(f"**Accuracy:** {accuracy:.2f}")

            # Save model
            model.save_model("pcos_model.txt")
            st.download_button("Download Model", "pcos_model.txt", "pcos_model.txt")

    except Exception as e:
        st.error(f"Error: {e}")

else:
    st.warning("Please upload a CSV file to proceed.")

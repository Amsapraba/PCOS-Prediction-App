import streamlit as st
import pandas as pd
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

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

        # Ensure column names are strings
        df.columns = df.columns.astype(str)

        # Drop any non-relevant columns if necessary (modify as needed)
        if 'ID' in df.columns:
            df.drop(columns=['ID'], inplace=True)

        # Identify target variable
        target_column = "PCOS (Y/N)"  # Change this if your target column has a different name

        if target_column not in df.columns:
            st.error("Target column not found in dataset!")
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

            # Split data into training and test sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Convert to LightGBM dataset format
            train_data = lgb.Dataset(X_train, label=y_train)
            valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

            # Define model parameters
            params = {
                "objective": "binary",
                "metric": "accuracy",
                "boosting_type": "gbdt",
                "learning_rate": 0.05,
                "num_leaves": 31,
                "verbose": -1
            }

            # Train the model
            model = lgb.train(params, train_data, valid_sets=[valid_data], verbose_eval=10)

            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_binary = np.round(y_pred)  # Convert probabilities to binary classes

            # Calculate accuracy
            accuracy = accuracy_score(y_test, y_pred_binary)

            # Display results
            st.subheader("Model Performance")
            st.write(f"**Accuracy:** {accuracy:.2f}")

    except Exception as e:
        st.error(f"Error loading file: {e}")

else:
    st.warning("Please upload a CSV file to proceed.")

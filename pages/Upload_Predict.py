import streamlit as st
import pandas as pd
from utils import load_model, predict
import os
from pathlib import Path
import joblib

def show():
    st.title("Upload Data & Predict")

    # Get available models
    models_dir = Path("models")
    if not models_dir.exists() or not any(models_dir.glob("*.pkl")):
        st.warning("No trained models found. Train a model first.")
        return

    model_files = [f for f in models_dir.glob("*.pkl")]
    selected_model = st.selectbox("Select a model", model_files)

    if selected_model:
        try:
            # Load model and get feature names
            model_data = joblib.load(selected_model)
            model = model_data['model']
            feature_names = model_data['feature_names']
            target_column = model_data['target_column']

            st.write("Required Features:", feature_names)
            st.write("Target Column:", target_column)

            # File uploader
            uploaded_file = st.file_uploader("Upload CSV for Prediction", type=["csv"])

            if uploaded_file:
                try:
                    # Read the file
                    df = pd.read_csv(uploaded_file)

                    # Display uploaded data
                    st.write("Uploaded Data Preview:")
                    st.dataframe(df.head())

                    # Check if all required features are present
                    missing_features = [f for f in feature_names if f not in df.columns]
                    if missing_features:
                        st.error(f"Missing required features: {missing_features}")
                        return

                    # Select only the required features in the correct order
                    df = df[feature_names]

                    # Make predictions
                    predictions = predict(model, df)

                    # Add predictions to the dataframe
                    df["Predictions"] = predictions

                    # Display results
                    st.write("Predictions:")
                    st.dataframe(df[["Predictions"]])

                    # Download button
                    st.download_button(
                        "Download Predictions",
                        df.to_csv(index=False).encode('utf-8'),
                        "predictions.csv",
                        "text/csv"
                    )

                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")
                    st.error("Please ensure your data matches the required format and features.")

        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            st.error("Please try selecting a different model or train a new one.")

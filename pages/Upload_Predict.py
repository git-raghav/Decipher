import streamlit as st
import pandas as pd
from utils import load_model, predict
import os

def show():
    st.title("Upload Data & Predict")

    model_files = [f for f in os.listdir("models") if f.endswith(".pkl")]

    if model_files:
        selected_model = st.selectbox("Select a model", model_files)
        uploaded_file = st.file_uploader("Upload CSV for Prediction", type=["csv"])
        
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.write("Uploaded Data:", df.head())

            model_path = f"models/{selected_model}"
            model = load_model(model_path)
            predictions = predict(model, df)
            
            st.write("Predictions:", predictions)
            df["Predictions"] = predictions
            st.download_button("Download Predictions", df.to_csv(index=False).encode('utf-8'), "predictions.csv")
    else:
        st.warning("No trained models found. Train a model first.")

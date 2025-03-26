import streamlit as st
import pandas as pd
import io
import os
from utils import train_and_save_model

def show():
    st.title("Train Your Model")
    
    # Add file size limit warning
    st.info("Maximum file size: 200MB. Supported format: CSV")
    
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
    
    if uploaded_file:
        try:
            # Debug information
            st.write("File details:")
            st.write(f"File name: {uploaded_file.name}")
            st.write(f"File size: {uploaded_file.size} bytes")
            
            # Read the file with explicit encoding handling
            try:
                df = pd.read_csv(io.StringIO(uploaded_file.getvalue().decode("utf-8")))
            except UnicodeDecodeError:
                # Try with different encoding if UTF-8 fails
                df = pd.read_csv(io.StringIO(uploaded_file.getvalue().decode("latin-1")))
            
            st.write("Dataset Preview:", df.head())
            st.write("Dataset Shape:", df.shape)
            st.write("Columns:", df.columns.tolist())

            target_column = st.selectbox("Select Target Column", df.columns)
            model_choice = st.selectbox("Select Model", ["Logistic Regression", "Random Forest", "SVM", "XGBoost"])

            if st.button("Train Model"):
                try:
                    os.makedirs("models", exist_ok=True)  # Ensure the models folder exists
                    accuracy, model_path = train_and_save_model(df, target_column, model_choice)
                    st.success(f"Model trained successfully! Accuracy: {accuracy:.2f}")
                    st.write(f"Model saved at: `{model_path}`")
                except Exception as e:
                    st.error(f"Error during model training: {str(e)}")
                    st.error("Please check if your data is properly formatted and the target column contains valid values.")

        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
            st.error("Please ensure your file is a valid CSV and not corrupted.")

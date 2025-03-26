import streamlit as st

def show():
    st.title("Welcome to ML Trainer & Analyzer")
    st.write("""
    This tool allows you to:
    - Train multiple ML models on your dataset
    - Save trained models for future use
    - Upload new data and get predictions
    - Analyze model performance using Evidently AI and SHAP
    """)

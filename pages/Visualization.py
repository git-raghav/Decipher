import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, ClassificationPreset
from utils import load_model
import os

def show():
    st.title("Visualization & Explainability")
    
    model_files = [f for f in os.listdir("models") if f.endswith(".pkl")]
    
    if model_files:
        selected_model = st.selectbox("Select a model", model_files)
        uploaded_file = st.file_uploader("Upload CSV for Analysis", type=["csv"])
        
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            model = load_model(f"models/{selected_model}")

            # SHAP Explanation
            explainer = shap.Explainer(model, df)
            shap_values = explainer(df)
            
            st.subheader("SHAP Summary Plot")
            fig, ax = plt.subplots(figsize=(10, 5))
            shap.summary_plot(shap_values, df, show=False)
            st.pyplot(fig)

            # Evidently AI Report
            st.subheader("Evidently AI Report")
            ref_data = df.sample(frac=0.5, random_state=42)
            report = Report(metrics=[DataDriftPreset(), ClassificationPreset()])
            report.run(reference_data=ref_data, current_data=df)
            st.write(report.show(mode="inline"))
    else:
        st.warning("No trained models found. Train a model first.")

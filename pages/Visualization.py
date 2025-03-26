import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from utils import load_model, get_model_info
import os
from pathlib import Path
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from scipy import stats
import joblib

def show():
    st.title("Model Visualization & Monitoring")

    # Get available models and datasets
    models_dir = Path("models")
    datasets_dir = Path("datasets")

    if not models_dir.exists() or not any(models_dir.glob("*.pkl")):
        st.warning("No trained models found. Please train a model first.")
        return

    if not datasets_dir.exists() or not any(datasets_dir.glob("*.csv")):
        st.warning("No datasets available. Please upload a dataset first.")
        return

    # Sidebar for model and dataset selection
    with st.sidebar:
        st.subheader("Select Model & Dataset")
        model_files = [f for f in models_dir.glob("*.pkl")]
        selected_model = st.selectbox(
            "Select Model",
            options=[f.name for f in model_files],
            format_func=lambda x: x.replace(".pkl", "").replace("_", " ").title()
        )

        dataset_files = [f for f in datasets_dir.glob("*.csv")]
        selected_dataset = st.selectbox(
            "Select Dataset",
            options=[f.name for f in dataset_files],
            format_func=lambda x: x.replace(".csv", "").title()
        )

    if selected_model and selected_dataset:
        try:
            # Load model and dataset
            model_data = joblib.load(models_dir / selected_model)
            model = model_data['model']
            feature_names = model_data['feature_names']
            target_column = model_data['target_column']

            df = pd.read_csv(datasets_dir / selected_dataset)

            # Get model information
            model_info = get_model_info(models_dir / selected_model)

            # Main content
            st.subheader("Model Information")
            col1, col2 = st.columns(2)
            with col1:
                st.write("Model Type:", model_info["type"])
                st.write("Parameters:", model_info["parameters"])
            with col2:
                st.write("Dataset Shape:", df.shape)
                st.write("Features:", len(df.columns))

            # Feature Importance
            if model_info["feature_importance"] is not None:
                st.subheader("Feature Importance")
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': model_info["feature_importance"]
                }).sort_values('Importance', ascending=False)

                fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h')
                st.plotly_chart(fig)

            # SHAP Values
            st.subheader("SHAP Values")
            try:
                X = df[feature_names]
                explainer = shap.Explainer(model, X)
                shap_values = explainer(X)

                # Summary plot
                fig, ax = plt.subplots(figsize=(10, 6))
                shap.summary_plot(shap_values, X, show=False)
                st.pyplot(fig)

                # Bar plot
                fig, ax = plt.subplots(figsize=(10, 6))
                shap.summary_plot(shap_values, X, plot_type="bar", show=False)
                st.pyplot(fig)
            except Exception as e:
                st.warning(f"Could not compute SHAP values: {str(e)}")

            # Data Quality Analysis
            st.subheader("Data Quality Analysis")
            col1, col2 = st.columns(2)
            with col1:
                st.write("Missing Values:")
                missing_values = df.isnull().sum()
                fig = px.bar(x=missing_values.index, y=missing_values.values)
                st.plotly_chart(fig)
            with col2:
                st.write("Data Types:")
                st.write(df.dtypes)

            # Feature Distributions
            st.subheader("Feature Distributions")
            for column in feature_names:
                fig = px.histogram(df, x=column, color=target_column, marginal="box")
                st.plotly_chart(fig)

            # Correlation Matrix
            st.subheader("Feature Correlation Matrix")
            corr_matrix = df[feature_names + [target_column]].corr()
            fig = px.imshow(corr_matrix, color_continuous_scale="RdBu")
            st.plotly_chart(fig)

            # Model Performance Metrics
            st.subheader("Model Performance Metrics")
            X = df[feature_names]
            y_true = df[target_column]
            y_pred = model.predict(X)

            # Classification Report
            report = classification_report(y_true, y_pred)
            st.text(report)

            # Confusion Matrix
            cm = confusion_matrix(y_true, y_pred)
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', ax=ax)
            st.pyplot(fig)

            # ROC Curve (if applicable)
            try:
                y_pred_proba = model.predict_proba(X)[:, 1]
                fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
                roc_auc = auc(fpr, tpr)

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC curve (AUC = {roc_auc:.2f})'))
                fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random'))
                fig.update_layout(title='ROC Curve', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
                st.plotly_chart(fig)
            except Exception as e:
                st.warning(f"Could not generate ROC curve: {str(e)}")

            # Statistical Tests
            st.subheader("Statistical Analysis")
            for column in feature_names:
                if df[column].dtype in ['int64', 'float64']:
                    # Perform t-test between feature and target
                    t_stat, p_value = stats.ttest_ind(df[df[target_column] == 0][column],
                                                    df[df[target_column] == 1][column])
                    st.write(f"{column}: t-statistic = {t_stat:.4f}, p-value = {p_value:.4f}")

        except Exception as e:
            st.error(f"Error processing model and data: {str(e)}")
            st.error("Please ensure the model and dataset are compatible.")

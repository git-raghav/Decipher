import streamlit as st
import pandas as pd
from pathlib import Path
import os
from utils import train_and_save_model
import plotly.express as px
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

def show():
    st.title("Train ML Models")

    # Get available datasets
    datasets_dir = Path("datasets")
    if not datasets_dir.exists():
        st.warning("No datasets available. Please upload a dataset first.")
        return

    available_datasets = [f for f in datasets_dir.glob("*.csv")]
    if not available_datasets:
        st.warning("No datasets available. Please upload a dataset first.")
        return

    # Dataset selection
    selected_dataset = st.selectbox(
        "Select Dataset",
        options=[f.name for f in available_datasets],
        format_func=lambda x: x.replace(".csv", "").title()
    )

    if selected_dataset:
        # Load the selected dataset
        df = pd.read_csv(datasets_dir / selected_dataset)

        # Display dataset info
        col1, col2 = st.columns(2)
        with col1:
            st.write("Dataset Shape:", df.shape)
        with col2:
            st.write("Number of Features:", len(df.columns))

        # Feature selection
        st.subheader("Feature Selection")
        target_column = st.selectbox("Select Target Column", df.columns)

        # Feature columns (excluding target)
        feature_columns = [col for col in df.columns if col != target_column]

        # Model selection
        st.subheader("Model Selection")
        model_choice = st.selectbox(
            "Select Model",
            ["Logistic Regression", "Random Forest", "SVM", "XGBoost"]
        )

        # Model parameters
        st.subheader("Model Parameters")
        if model_choice == "Random Forest":
            n_estimators = st.slider("Number of Trees", 10, 200, 100)
            max_depth = st.slider("Max Depth", 2, 20, 5)
        elif model_choice == "XGBoost":
            n_estimators = st.slider("Number of Trees", 10, 200, 100)
            learning_rate = st.slider("Learning Rate", 0.01, 0.3, 0.1)
        elif model_choice == "SVM":
            kernel = st.selectbox("Kernel", ["rbf", "linear", "poly"])
            C = st.slider("C (Regularization)", 0.1, 10.0, 1.0)

        # Training options
        st.subheader("Training Options")
        test_size = st.slider("Test Set Size", 0.1, 0.4, 0.2, 0.05)
        random_state = st.number_input("Random State", 0, 1000, 42)

        if st.button("Train Model"):
            try:
                # Create models directory if it doesn't exist
                models_dir = Path("models")
                models_dir.mkdir(exist_ok=True)

                # Train model and get metrics
                accuracy, model_path, y_test, y_pred, feature_importance = train_and_save_model(
                    df, target_column, model_choice,
                    test_size=test_size,
                    random_state=random_state
                )

                # Display results
                st.success(f"Model trained successfully! Accuracy: {accuracy:.2f}")
                st.write(f"Model saved at: `{model_path}`")

                # Display confusion matrix
                st.subheader("Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', ax=ax)
                st.pyplot(fig)

                # Display classification report
                st.subheader("Classification Report")
                report = classification_report(y_test, y_pred)
                st.text(report)

                # Display feature importance if available
                if feature_importance is not None:
                    st.subheader("Feature Importance")
                    importance_df = pd.DataFrame({
                        'Feature': feature_columns,
                        'Importance': feature_importance
                    }).sort_values('Importance', ascending=False)

                    fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h')
                    st.plotly_chart(fig)

            except Exception as e:
                st.error(f"Error during model training: {str(e)}")
                st.error("Please check if your data is properly formatted and the target column contains valid values.")

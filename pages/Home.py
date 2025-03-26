import streamlit as st

def show():
    st.title("Decipher")

    st.markdown("""
    ## Welcome to ML Model Trainer & Analyzer

    This comprehensive tool helps you train, analyze, and deploy machine learning models with ease.

    ### Key Features:

    #### 📊 Dataset Management
    - Load and store preprocessed datasets (Titanic, Iris, Mushrooms, etc.)
    - Persistent storage of datasets for future use
    - Easy dataset selection and management

    #### 🤖 Model Training
    - Train multiple ML models on your dataset
    - Support for various algorithms:
        - Logistic Regression
        - Random Forest
        - Support Vector Machine (SVM)
        - XGBoost
    - Automatic model saving and versioning

    #### 🔮 Predictions
    - Make predictions on new data
    - Download prediction results
    - Batch prediction support

    #### 📈 Visualization & Analysis
    - Comprehensive model performance metrics
    - SHAP values for feature importance
    - Evidently AI for model monitoring
    - Data drift detection
    - Interactive visualizations

    ### Getting Started:
    1. Go to the "Dataset Load" page to upload or select a dataset
    2. Navigate to "Train Models" to train your ML models
    3. Use "Predictions" to make predictions on new data
    4. Visit "Visualization" for detailed model analysis

    """)

    # Add a footer with contact information
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>Built with ❤️ using Streamlit</p>
        <p>For support or questions, please contact us</p>
    </div>
    """, unsafe_allow_html=True)

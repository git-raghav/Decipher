import streamlit as st
from pages import Welcome, Dataset_Load, Train_Models, Upload_Predict, Visualization

# Set page config
st.set_page_config(
    page_title="ML Model Trainer & Analyzer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main content
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

# Show the welcome page
Welcome.show()

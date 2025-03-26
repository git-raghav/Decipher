import streamlit as st
from pages import Home, Train_Models, Upload_Predict, Visualization

st.set_page_config(page_title="ML Trainer & Analyzer", layout="wide")

# Set Home as the default landing page
if "page" not in st.session_state:
    st.session_state["page"] = "Home"

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Train Models", "Upload & Predict", "Visualization"], index=0)

# Store the selected page in session state
st.session_state["page"] = page

# Render the selected page
if st.session_state["page"] == "Home":
    Home.show()
elif st.session_state["page"] == "Train Models":
    Train_Models.show()
elif st.session_state["page"] == "Upload & Predict":
    Upload_Predict.show()
elif st.session_state["page"] == "Visualization":
    Visualization.show()

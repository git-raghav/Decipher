import streamlit as st
from pages import Home, Dataset_Load, Train_Models, Upload_Predict, Visualization

# Set page config
st.set_page_config(
    page_title="Decipher",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set Home as the default landing page
if "page" not in st.session_state:
    st.session_state["page"] = "Home"

# Sidebar navigation
with st.sidebar:
    st.title("🤖 Decipher")
    st.markdown("---")

    # Navigation
    st.subheader("Navigation")
    page = st.radio(
        "Go to",
        ["Home", "Dataset Load", "Train Models", "Upload & Predict", "Visualization"],
        index=0
    )

    # Store the selected page in session state
    st.session_state["page"] = page

    # Add some helpful information
    st.markdown("---")
    st.markdown("""
    ### Quick Tips
    - Start by loading or selecting a dataset
    - Train your model with the selected dataset
    - Make predictions on new data
    - Visualize and monitor model performance
    """)

    # Add footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>Built with ❤️ using Streamlit</p>
        <p>Version 1.0.0</p>
    </div>
    """, unsafe_allow_html=True)

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

# Render the selected page
if st.session_state["page"] == "Home":
    Home.show()
elif st.session_state["page"] == "Dataset Load":
    Dataset_Load.show()
elif st.session_state["page"] == "Train Models":
    Train_Models.show()
elif st.session_state["page"] == "Upload & Predict":
    Upload_Predict.show()
elif st.session_state["page"] == "Visualization":
    Visualization.show()

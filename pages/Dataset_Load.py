import streamlit as st
import pandas as pd
import os
from pathlib import Path

def show():
    # Initialize session state for refresh
    if 'refresh' not in st.session_state:
        st.session_state.refresh = False

    st.title("Dataset Management")

    # Create datasets directory if it doesn't exist
    datasets_dir = Path("datasets")
    datasets_dir.mkdir(exist_ok=True)

    # List existing datasets
    existing_datasets = [f for f in datasets_dir.glob("*.csv")]

    # Sidebar for dataset selection
    with st.sidebar:
        st.subheader("Available Datasets")
        if existing_datasets:
            selected_dataset = st.selectbox(
                "Select a dataset",
                options=[f.name for f in existing_datasets],
                format_func=lambda x: x.replace(".csv", "")
            )
            if selected_dataset:
                df = pd.read_csv(datasets_dir / selected_dataset)
                st.write(f"Shape: {df.shape}")
                st.write("Columns:", df.columns.tolist())
        else:
            st.info("No datasets available. Upload a new dataset below.")

    # Main content
    st.subheader("Upload New Preprocessed Dataset")

    # File uploader
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file:
        try:
            # Read the file
            df = pd.read_csv(uploaded_file)

            # Display dataset info
            st.write("### Dataset Preview")
            st.dataframe(df.head())

            st.write("### Dataset Information")
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"Shape: {df.shape}")
                st.write(f"Number of columns: {len(df.columns)}")
            with col2:
                st.write("Data Types:")
                st.write(df.dtypes)

            # Save dataset
            if st.button("Save Dataset"):
                # Create a clean filename
                filename = uploaded_file.name.lower().replace(" ", "_")
                if not filename.endswith(".csv"):
                    filename += ".csv"

                # Save to datasets directory
                filepath = datasets_dir / filename
                df.to_csv(filepath, index=False)
                st.success(f"Dataset saved successfully as {filename}")
                st.rerun()

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

    # Add some sample datasets
    st.subheader("Sample Datasets")
    sample_datasets = {
        "Titanic": "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv",
        "Iris": "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
    }

    for name, url in sample_datasets.items():
        if st.button(f"Load {name} Dataset"):
            try:
                df = pd.read_csv(url)
                filename = f"{name.lower()}.csv"
                filepath = datasets_dir / filename
                df.to_csv(filepath, index=False)
                st.success(f"{name} dataset loaded and saved successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Error loading {name} dataset: {str(e)}")

    # Dataset management section
    if existing_datasets:
        st.subheader("Dataset Management")
        for dataset in existing_datasets:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"**{dataset.stem}**")
                df = pd.read_csv(dataset)
                st.write(f"Shape: {df.shape} | Columns: {len(df.columns)}")
            with col2:
                if st.button("Delete", key=f"delete_{dataset.stem}"):
                    try:
                        os.remove(dataset)
                        st.success(f"Deleted {dataset.name}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error deleting {dataset.name}: {str(e)}")

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import numpy as np
from pathlib import Path

MODELS = {
    "Logistic Regression": LogisticRegression,
    "Random Forest": RandomForestClassifier,
    "SVM": SVC,
    "XGBoost": XGBClassifier
}

import os

def train_and_save_model(df, target_column, model_name, test_size=0.2, random_state=42):
    """
    Train a model and save it to disk.

    Args:
        df: DataFrame containing the data
        target_column: Name of the target column
        model_name: Name of the model to train
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility

    Returns:
        tuple: (accuracy, model_path, y_test, y_pred, feature_importance)
    """
    # Prepare data
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Initialize and train model
    model_class = MODELS[model_name]
    model = model_class()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Get feature importance if available
    feature_importance = None
    if hasattr(model, 'feature_importances_'):
        feature_importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        feature_importance = np.abs(model.coef_[0])

    # Save model and feature names
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    model_filename = f"models/{model_name.replace(' ', '_')}_{random_state}.pkl"

    # Create a dictionary containing both the model and feature names
    model_data = {
        'model': model,
        'feature_names': X.columns.tolist(),
        'target_column': target_column
    }

    joblib.dump(model_data, model_filename)

    return accuracy, model_filename, y_test, y_pred, feature_importance

def load_model(model_path):
    """Load a trained model from disk."""
    model_data = joblib.load(model_path)
    return model_data['model']

def predict(model, df):
    """Make predictions using a trained model."""
    return model.predict(df)

def get_model_info(model_path):
    """Get information about a trained model."""
    model_data = joblib.load(model_path)
    model = model_data['model']
    feature_names = model_data['feature_names']
    target_column = model_data['target_column']

    info = {
        "type": type(model).__name__,
        "parameters": model.get_params(),
        "feature_importance": None,
        "feature_names": feature_names,
        "target_column": target_column
    }

    if hasattr(model, 'feature_importances_'):
        info["feature_importance"] = model.feature_importances_
    elif hasattr(model, 'coef_'):
        info["feature_importance"] = np.abs(model.coef_[0])

    return info

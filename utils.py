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

    # Save model
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    model_filename = f"models/{model_name.replace(' ', '_')}_{random_state}.pkl"
    joblib.dump(model, model_filename)

    return accuracy, model_filename, y_test, y_pred, feature_importance

def load_model(model_path):
    """Load a trained model from disk."""
    return joblib.load(model_path)

def predict(model, df):
    """Make predictions using a trained model."""
    return model.predict(df)

def get_model_info(model_path):
    """Get information about a trained model."""
    model = load_model(model_path)
    info = {
        "type": type(model).__name__,
        "parameters": model.get_params(),
        "feature_importance": None
    }

    if hasattr(model, 'feature_importances_'):
        info["feature_importance"] = model.feature_importances_
    elif hasattr(model, 'coef_'):
        info["feature_importance"] = np.abs(model.coef_[0])

    return info

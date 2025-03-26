import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

MODELS = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(),
    "XGBoost": XGBClassifier()
}

import os

def train_and_save_model(df, target_column, model_name):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = MODELS[model_name]
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # **Ensure models directory exists**
    os.makedirs("models", exist_ok=True)  
    model_filename = f"models/{model_name.replace(' ', '_')}.pkl"  # Avoid spaces in filename
    joblib.dump(model, model_filename)

    return accuracy, model_filename

def load_model(model_path):
    return joblib.load(model_path)

def predict(model, df):
    return model.predict(df)

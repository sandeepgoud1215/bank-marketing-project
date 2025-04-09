# src/predict.py

import joblib

def load_model(model_path="model.pkl"):
    return joblib.load(model_path)

def make_prediction(model, X):
    return model.predict(X)

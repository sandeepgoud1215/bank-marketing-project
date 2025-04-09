# app/app.py

import streamlit as st
import pandas as pd
import pickle
from utils.helper import make_prediction

# Load model, scaler, and encoder
with open('model/model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('encoder/encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)

st.title("Bank Marketing Campaign Prediction")
st.markdown("This app predicts whether a client will subscribe to a term deposit.")

# Collect user input
def user_input():
    age = st.slider("Age", 18, 95, 30)
    job = st.selectbox("Job", ['admin.', 'technician', 'services', 'management', 'retired', 'blue-collar', 'unemployed', 'entrepreneur', 'housemaid', 'student', 'self-employed'])
    marital = st.selectbox("Marital", ['married', 'single', 'divorced'])
    education = st.selectbox("Education", ['secondary', 'tertiary', 'primary', 'unknown'])
    default = st.selectbox("Credit Default", ['yes', 'no'])
    balance = st.number_input("Balance", -2000, 100000, 1000)
    housing = st.selectbox("Housing Loan", ['yes', 'no'])
    loan = st.selectbox("Personal Loan", ['yes', 'no'])

    data = {
        'age': age,
        'job': job,
        'marital': marital,
        'education': education,
        'default': default,
        'balance': balance,
        'housing': housing,
        'loan': loan
    }
    return pd.DataFrame([data])

df = user_input()

if st.button("Predict"):
    result = make_prediction(df, model, encoder, scaler)
    st.success(f"Prediction: {result}")

import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
import json

st.title("Breast Cancer Prediction App (Logistic Regression)")

# Load saved components
model = load("model.joblib")
scaler = load("scaler.joblib")

with open("feature_names.json", "r") as f:
    feature_names = json.load(f)

st.write("### Enter Input Values")

inputs = {}
for col in feature_names:
    inputs[col] = st.number_input(col, value=0.0)

if st.button("Predict"):
    df = pd.DataFrame([inputs])
    scaled = scaler.transform(df)
    prediction = model.predict(scaled)[0]
    result = "Malignant (Cancer Detected)" if prediction == 1 else "Benign (No Cancer)"
    st.success(result)

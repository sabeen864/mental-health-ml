import streamlit as st
import pandas as pd
import requests

st.set_page_config(page_title="Mental Health Predictor", layout="centered")

st.title("🧠 Mental Health in Tech Predictor")
st.markdown("Predict whether someone needs mental health treatment based on simple inputs.")

# User Inputs
Age = st.slider("Age", 18, 80, 30)
Gender = st.selectbox("Gender", ["Male", "Female", "Other"])
family_history = st.radio("Family History of Mental Illness?", ["Yes", "No"])
remote_work = st.radio("Do you work remotely?", ["Yes", "No"])
self_employed = st.radio("Are you self-employed?", ["Yes", "No"])
benefits = st.radio("Does your employer provide mental health benefits?", ["Yes", "No"])
care_options = st.radio("Does your employer offer mental health care options?", ["Yes", "No"])
anonymity = st.radio("Is anonymity protected when seeking help?", ["Yes", "No"])
leave = st.radio("Ease of taking medical leave?", ["Never", "Rarely", "Sometimes", "Often"])
work_interfere = st.radio("How often does your work interfere with your mental health?", ["Never", "Rarely", "Sometimes", "Often"])

if st.button("🔍 Predict"):
    input_data = {
        "Age": Age,
        "Gender": Gender,
        "family_history": family_history,
        "remote_work": remote_work,
        "self_employed": self_employed,
        "benefits": benefits,
        "care_options": care_options,
        "anonymity": anonymity,
        "leave": leave,
        "work_interfere": work_interfere
    }

    try:
        res = requests.post(
            "https://mental-health-ml-api-ctexdderg6asfpf3.canadacentral-01.azurewebsites.net/predict",
            json=input_data
        )

        if res.status_code == 200:
            prediction = res.json()['prediction']
            st.success(f"Prediction: {prediction}")
        else:
            st.error(f"API error: {res.text}")
    except Exception as e:
        st.error(f"Connection Error: {e}")
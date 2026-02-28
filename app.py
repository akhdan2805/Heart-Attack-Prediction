import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model
model = joblib.load("artifacts/model.pkl")

def main():
    st.title('Machine Learning Heart Attack Prediction Model Deployment')

    # Add user input components
    age = st.number_input('Input Age', min_value=20, max_value=80, value=20)

    sex_option = st.selectbox("Choose Gender", ["Female", "Male"])
    sex = 1 if sex_option == "Male" else 0

    cp = st.selectbox("Chestpain", [0, 1, 2, 3])
    trestbps = st.number_input('Blood Pressure', min_value=90, max_value=200, value=90)
    chol = st.number_input('Cholestrol', min_value=120, max_value=570, value=120)
    fbs = st.selectbox("Bloodsugar", [0, 1])
    restecg = st.selectbox("ECG Result", [0, 1, 2])
    thalach = st.number_input('Max Heart Rate', min_value=70, max_value=205, value=70)
    exang = st.selectbox("Exercise Angina", [0, 1])
    oldpeak = st.number_input('ST Depression', min_value=0.0, max_value=10.0, value=0.0)
    slope = st.selectbox("ST Slope", [0, 1, 2])
    ca = st.selectbox("Major Vessels", [0, 1, 2, 3, 4])
    thal = st.selectbox("Thalassemia", [0, 1, 2, 3])
    
    if st.button('Make Prediction'):
        features = [age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]
        result = make_prediction(features)
        if result == 1:
            prediction_text = "High Risk of Heart Attack"
            st.error(f"Result: {prediction_text}")
        else:
            prediction_text = "Low/Normal Risk of Heart Attack"
            st.success(f"Result: {prediction_text}")

def make_prediction(features):
    # Use the loaded model to make predictions
    input_array = np.array(features, dtype=float).reshape(1, -1)
    prediction = model.predict(input_array)
    return prediction[0]

if __name__ == '__main__':
    main()
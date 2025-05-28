import streamlit as st
import joblib
import numpy as np

st.title("Diabetes Progression Predictor by Ashish")

# Load model
model = joblib.load("model.pkl")

st.subheader("Enter Patient Features")

# Create 10 separate input fields
age = st.text_input("Age")
sex = st.text_input("Sex")
bmi = st.text_input("BMI")
bp = st.text_input("Blood Pressure")
s1 = st.text_input("S1")
s2 = st.text_input("S2")
s3 = st.text_input("S3")
s4 = st.text_input("S4")
s5 = st.text_input("S5")
s6 = st.text_input("S6")

if st.button("Predict Progression"):
    try:
        # Convert input to float and form feature array
        features = [float(age), float(sex), float(bmi), float(bp), float(s1), float(s2), float(s3), float(s4), float(s5), float(s6)]
        features = np.array(features).reshape(1, -1)
        prediction = model.predict(features)
        st.success(f"Predicted Disease Progression: {prediction[0]:.2f}")
    except Exception as e:
        st.error(f"Invalid input: {e}")

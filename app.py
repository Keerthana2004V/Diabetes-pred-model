
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load the trained model
model = load_model("diabetes_model.keras")

# Title
st.title("OptiANN-LR: Diabetes Prediction App")
st.markdown("### Enter health information to predict diabetes:")

# User inputs
pregnancies = st.number_input("Pregnancies", min_value=0)
glucose = st.number_input("Glucose", min_value=0)
bp = st.number_input("Blood Pressure", min_value=0)
skin = st.number_input("Skin Thickness", min_value=0)
insulin = st.number_input("Insulin", min_value=0)
bmi = st.number_input("BMI", min_value=0.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0)
age = st.number_input("Age", min_value=0)

# Predict button
if st.button("Predict"):
    # Input data as array
    user_data = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])

    # Fit the scaler on original data (same as training)
    df = pd.read_csv("https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv", 
                      names=["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", 
                            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"])
    X = df.drop("Outcome", axis=1)
    scaler = StandardScaler()
    scaler.fit(X)

    # Scale user input
    user_data_scaled = scaler.transform(user_data)

    # Predict
    prediction = model.predict(user_data_scaled)[0][0]

    # Output
    if prediction > 0.5:
        st.error("Diabetic")
    else:
        st.success("Not Diabetic")

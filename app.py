import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load trained model and scaler
model = joblib.load('diabetes_rf_model_top.joblib')
scaler = joblib.load('scaler_top.joblib')

# Feature list used for training
top_features = ['GenHlth', 'BMI', 'Age', 'Income', 'Education', 'PhysHlth', 'MentHlth', 'HighBP']

# UI inputs
st.title("Diabetes Risk Prediction")

st.markdown("Provide your health details to assess diabetes risk.")

genhlth = st.slider("General Health (1=Excellent to 5=Poor)", 1, 5, 3)
bmi = st.number_input("BMI (Body Mass Index)", 10.0, 70.0, 25.0)
age = st.slider("Age Group (1=18-24 to 13=80+)", 1, 13, 5)
income = st.slider("Income Group (1=Low to 8=High)", 1, 8, 4)
education = st.slider("Education Level (1=Less than 9th Grade to 6=College Grad)", 1, 6, 4)
physhlth = st.slider("Bad Physical Health Days (last 30 days)", 0, 30, 5)
menthlth = st.slider("Poor Mental Health Days (last 30 days)", 0, 30, 5)
highbp = st.selectbox("High Blood Pressure", ["No", "Yes"])
highbp = 1 if highbp == "Yes" else 0

# Fixed threshold at 0.5 (default)
FIXED_THRESHOLD = 0.5

# Button to predict
if st.button("Predict Risk"):
    # Prepare input DataFrame with correct feature names
    input_df = pd.DataFrame([[genhlth, bmi, age, income, education, physhlth, menthlth, highbp]], columns=top_features)

    # Scale the input
    input_scaled = scaler.transform(input_df)

    # Predict probability of diabetes risk (positive class)
    prob = model.predict_proba(input_scaled)[0][1]

    # Determine prediction based on fixed threshold 0.5
    if prob >= FIXED_THRESHOLD:
        prediction = "High Risk of Diabetes"
        confidence = prob
    elif prob >= FIXED_THRESHOLD - 0.1:  # Moderate risk zone just below threshold
        prediction = "Moderate Risk of Diabetes"
        confidence = prob
    else:
        prediction = "Low Risk of Diabetes"
        confidence = 1 - prob

    st.markdown(f"### **{prediction}** (Confidence: `{confidence:.2f}`)")

    # Show warnings based on input values
    # BMI warnings
    if bmi > 40:
        st.error("Very high BMI (Obesity class III)")
    elif bmi > 35:
        st.warning("High BMI (Obesity class II)")
    elif bmi > 30:
        st.info("Elevated BMI (Obesity class I)")

    # High blood pressure warning
    if highbp == 1:
        st.warning("You have High Blood Pressure")

    # General health warning
    if genhlth >= 4:
        st.warning("Self-reported general health is poor")

    # Confidence borderline info
    if abs(prob - FIXED_THRESHOLD) <= 0.05:
        st.info("This prediction is close to the threshold. Consider medical testing for a definitive diagnosis.")

    # Optionally show scaled input or raw data for debugging
    with st.expander("View Scaled Input (Debug)"):
        st.write(pd.DataFrame(input_scaled, columns=top_features))

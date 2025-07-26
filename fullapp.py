import streamlit as st
import pandas as pd
import joblib

# Load the trained model and feature names
model = joblib.load("revised_diabetes_rf_model.joblib")
model_features = joblib.load("revised_model_features.joblib")

# Page config
st.set_page_config(page_title="Diabetes Prediction App", layout="wide")
st.title("Am I at Risk? – Diabetes Risk Prediction Application")

# --- User Input Form ---
with st.form("input_form"):
    st.subheader("Enter your health and lifestyle details")

    # Arrange input fields in three columns
    col1, col2, col3 = st.columns(3)

    with col1:
        HighBP = st.selectbox('High Blood Pressure (0: No, 1: Yes)', [0, 1])
        HighChol = st.selectbox('High Cholesterol (0: No, 1: Yes)', [0, 1])
        CholCheck = st.selectbox('Cholesterol Checked in last 5 years (0: No, 1: Yes)', [0, 1])
        BMI = st.slider('Your Body Mass Index (BMI)', 12.0, 98.0, 28.0)
        Smoker = st.selectbox('Have you smoked at least 100 cigarettes? (0: No, 1: Yes)', [0, 1])
        Stroke = st.selectbox('Ever had a stroke? (0: No, 1: Yes)', [0, 1])
        HeartDiseaseorAttack = st.selectbox('History of Heart Disease or Attack (0: No, 1: Yes)', [0, 1])

    with col2:
        PhysActivity = st.selectbox('Physical activity in past 30 days (0: No, 1: Yes)', [0, 1])
        Fruits = st.selectbox('Consume fruit 1 or more times per day (0: No, 1: Yes)', [0, 1])
        Veggies = st.selectbox('Consume vegetables 1 or more times per day (0: No, 1: Yes)', [0, 1])
        HvyAlcoholConsump = st.selectbox('Heavy alcohol consumption (0: No, 1: Yes)', [0, 1])
        AnyHealthcare = st.selectbox('Have any kind of health care coverage (0: No, 1: Yes)', [0, 1])
        NoDocbcCost = st.selectbox('Could not see a doctor due to cost (0: No, 1: Yes)', [0, 1])
        DiffWalk = st.selectbox('Have serious difficulty walking or climbing stairs (0: No, 1: Yes)', [0, 1])

    with col3:
        GenHlth = st.slider('General Health (1: Excellent to 5: Poor)', 1, 5, 3)
        MentHlth = st.slider('Days of poor mental health in last 30 days', 0, 30, 2)
        PhysHlth = st.slider('Days of physical illness in last 30 days', 0, 30, 3)
        Sex = st.selectbox('Sex (0: Female, 1: Male)', [0, 1])
        Age = st.slider('Age (1: 18-24 to 13: 80+)', 1, 13, 9)
        Education = st.slider('Education Level (1: Never attended to 6: College graduate)', 1, 6, 4)
        Income = st.slider('Income Level (1: <$10k to 8: >$75k)', 1, 8, 6)

    submitted = st.form_submit_button("Predict Diabetes Risk")

# --- Prediction Logic ---
if submitted:
    # Collect inputs into dictionary
    input_data = {
        'HighBP': float(HighBP), 'HighChol': float(HighChol), 'CholCheck': float(CholCheck),
        'BMI': float(BMI), 'Smoker': float(Smoker), 'Stroke': float(Stroke),
        'HeartDiseaseorAttack': float(HeartDiseaseorAttack), 'PhysActivity': float(PhysActivity),
        'Fruits': float(Fruits), 'Veggies': float(Veggies), 'HvyAlcoholConsump': float(HvyAlcoholConsump),
        'AnyHealthcare': float(AnyHealthcare), 'NoDocbcCost': float(NoDocbcCost),
        'GenHlth': float(GenHlth), 'MentHlth': float(MentHlth), 'PhysHlth': float(PhysHlth),
        'DiffWalk': float(DiffWalk), 'Sex': float(Sex), 'Age': float(Age),
        'Education': float(Education), 'Income': float(Income)
    }

    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])

    # Feature Engineering
    risk_factors = ['HighBP', 'HighChol', 'Smoker', 'Stroke', 'HeartDiseaseorAttack', 'HvyAlcoholConsump', 'DiffWalk']
    input_df['RiskFactorCount'] = input_df[risk_factors].sum(axis=1)

    input_df['Age_group_code'] = input_df['Age']
    input_df['BMI_category_code'] = pd.cut(input_df['BMI'],
                                           bins=[0, 18.5, 25, 30, 100],
                                           labels=[0, 1, 2, 3],
                                           include_lowest=True).astype(int)

    input_final = input_df.reindex(columns=model_features, fill_value=0)

    # Make predictions
    prediction = model.predict(input_final)[0]
    probability = model.predict_proba(input_final)[0][1]

    # Display result
    st.markdown("---")
    st.subheader("Prediction Result")

    if probability >= 0.4:
        st.error(f"High Risk of Diabetes (Confidence: {probability:.2%})")
    elif probability >= 0.3:
        st.warning(f"Moderate Risk of Diabetes (Confidence: {probability:.2%})")
    else:
        st.success(f"Low Risk of Diabetes (Confidence: {probability:.2%})")

   # st.markdown(f"**Predicted Outcome:** {'**Diabetic**' if prediction == 1 else '**Non-Diabetic**'}")

    st.caption("""
    **Disclaimer:** This prediction is based on a machine learning model and is not a substitute for professional medical advice.
    The confidence score represents the model's certainty in its prediction.
    """)

    # Optional Details
    with st.expander("View Feature Importances"):
        feat_imp_df = pd.DataFrame({
            "Feature": model_features,
            "Importance": model.feature_importances_
        }).sort_values(by="Importance", ascending=False).head(10)
        st.dataframe(feat_imp_df)

    with st.expander("View Your Input & Engineered Features"):
        st.dataframe(input_final)


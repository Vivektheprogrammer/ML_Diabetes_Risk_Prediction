import gradio as gr
import pandas as pd
import joblib

# Load model and features
model = joblib.load("revised_diabetes_rf_model.joblib")
model_features = joblib.load("revised_model_features.joblib")

# Prediction function
def predict_diabetes(
    HighBP, HighChol, CholCheck, BMI, Smoker, Stroke, HeartDiseaseorAttack,
    PhysActivity, Fruits, Veggies, HvyAlcoholConsump, AnyHealthcare, NoDocbcCost,
    DiffWalk, GenHlth, MentHlth, PhysHlth, Sex, Age, Education, Income
):
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

    # Prediction
    prediction = model.predict(input_final)[0]
    probability = model.predict_proba(input_final)[0][1]

    # Risk interpretation
    if probability >= 0.4:
        risk = f"**High Risk of Diabetes** (Confidence: {probability:.2%})"
    elif probability >= 0.3:
        risk = f"**Moderate Risk of Diabetes** (Confidence: {probability:.2%})"
    else:
        risk = f"**Low Risk of Diabetes** (Confidence: {probability:.2%})"

    # Feature importances
    feat_imp_df = pd.DataFrame({
        "Feature": model_features,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False).head(10)

    return risk, feat_imp_df, input_final

# Gradio inputs with instructions
inputs = [
    gr.Dropdown([0, 1], label="High Blood Pressure", info="Have you been diagnosed with high BP? (0: No, 1: Yes)"),
    gr.Dropdown([0, 1], label="High Cholesterol", info="Do you have high cholesterol?"),
    gr.Dropdown([0, 1], label="Cholesterol Checked", info="Was your cholesterol checked in last 5 years?"),
    gr.Slider(12.0, 98.0, value=28.0, label="BMI", info="Enter your Body Mass Index (BMI)"),
    gr.Dropdown([0, 1], label="Smoked 100 Cigarettes", info="Have you smoked at least 100 cigarettes in your life?"),
    gr.Dropdown([0, 1], label="Stroke", info="Have you ever had a stroke?"),
    gr.Dropdown([0, 1], label="Heart Disease or Attack", info="Do you have history of heart disease or heart attack?"),
    gr.Dropdown([0, 1], label="Physical Activity", info="Any physical activity in past 30 days?"),
    gr.Dropdown([0, 1], label="Fruits Daily", info="Do you eat fruits at least once daily?"),
    gr.Dropdown([0, 1], label="Vegetables Daily", info="Do you eat vegetables at least once daily?"),
    gr.Dropdown([0, 1], label="Heavy Alcohol", info="Do you consume alcohol heavily (men >14/week, women >7/week)?"),
    gr.Dropdown([0, 1], label="Any Healthcare Coverage", info="Do you have any form of health coverage?"),
    gr.Dropdown([0, 1], label="Couldn't Afford Doctor", info="Did cost stop you from seeing a doctor in last year?"),
    gr.Dropdown([0, 1], label="Difficulty Walking", info="Do you have serious difficulty walking/climbing stairs?"),
    gr.Slider(1, 5, value=3, label="General Health (1–5)", info="1: Excellent, 5: Poor"),
    gr.Slider(0, 30, value=2, label="Poor Mental Health Days", info="How many days in last 30 did you feel mentally unwell?"),
    gr.Slider(0, 30, value=3, label="Poor Physical Health Days", info="How many days in last 30 did you feel physically ill?"),
    gr.Dropdown([0, 1], label="Sex (0: Female, 1: Male)", info="Your biological sex"),
    gr.Slider(1, 13, value=9, label="Age Group (1–13)", info="1: 18–24, 2: 25–29, ..., 13: 80+"),
    gr.Slider(1, 6, value=4, label="Education Level (1–6)", info="1: No schooling, 6: College graduate"),
    gr.Slider(1, 8, value=6, label="Income Level (1–8)", info="1: <rs20k, 8: >rs85k")
]

# Outputs
outputs = [
    gr.Markdown(label="Diabetes Risk Result"),
    gr.Dataframe(label="Top Feature Importances"),
    gr.Dataframe(label="Your Final Input Features")
]

# Interface
app = gr.Interface(
    fn=predict_diabetes,
    inputs=inputs,
    outputs=outputs,
    title="Am I at Risk?  Diabetes Prediction App",
    description=(
        "This app predicts your risk of developing diabetes based on health and lifestyle factors using a trained machine learning model. "
        "Please fill out all fields as accurately as possible."
    ),
    theme="default",  
    flagging_mode="never"
)

app.launch()

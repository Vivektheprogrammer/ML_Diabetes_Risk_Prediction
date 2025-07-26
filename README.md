# Diabetes Risk Prediction Application

A machine learning-powered web application that predicts diabetes risk based on health and lifestyle factors using the BRFSS 2015 dataset.

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)

## Live Demo

**Try the live application**: [Diabetes Risk Predictor](https://vivektheprogrammer-diabetes-risk-prediction-fullapp-h3k2ri.streamlit.app/)

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Model Performance](#model-performance)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Details](#model-details)
- [Deployment](#deployment)

## Overview

This application uses machine learning to assess diabetes risk based on various health indicators from the Behavioral Risk Factor Surveillance System (BRFSS) 2015 dataset. The model considers 21 different health and lifestyle factors to provide personalized risk assessments.

### Key Highlights
- **Interactive Web Interface**: User-friendly Streamlit application
- **Real-time Predictions**: Instant risk assessment with confidence scores
- **Feature Engineering**: Advanced feature extraction for improved accuracy
- **Model Interpretability**: Feature importance analysis and SHAP explanations

## Features

- **Risk Assessment**: Categorizes users into Low, Moderate, or High risk groups
- **Confidence Scoring**: Provides probability scores with precise percentages
- **24 Health Indicators**: Comprehensive health and lifestyle assessment
- **Feature Importance**: Interactive visualization of top contributing factors
- **Responsive Design**: Clean, user-friendly interface with three-column layout
- **Real-time Predictions**: Instant risk assessment upon form submission
- **Model Transparency**: View your input data and engineered features
- **Educational Content**: Detailed explanations and medical disclaimers

## Dataset

The model is trained on the **Diabetes Health Indicators Dataset** from BRFSS 2015, which includes:

- **253,680 survey responses** (original dataset)
- **70,692 balanced samples** (after SMOTE balancing)
- **24 feature variables** (21 original + 3 engineered)
- **Binary classification** (Diabetic vs Non-Diabetic)

### Key Features Used:
- **Health Conditions**: High BP, High Cholesterol, Stroke, Heart Disease
- **Lifestyle Factors**: Smoking, Physical Activity, Diet, Alcohol Consumption
- **Demographics**: Age, Sex, Education, Income
- **Physical Metrics**: BMI, General Health Status
- **Healthcare Access**: Insurance coverage, Cost barriers
- **Engineered Features**: Risk Factor Count, Age Group Code, BMI Category Code

### Top 10 Most Important Features:
1. **General Health** (13.53%) - Overall health self-assessment
2. **BMI** (11.53%) - Body Mass Index
3. **Risk Factor Count** (10.14%) - Aggregated risk factors
4. **High Blood Pressure** (8.63%) - Hypertension status
5. **Age** (7.07%) - Age demographic
6. **Age Group Code** (6.61%) - Categorical age grouping
7. **Income** (5.94%) - Income level
8. **Physical Health** (5.31%) - Days of poor physical health
9. **Education** (4.00%) - Education level
10. **Mental Health** (3.89%) - Days of poor mental health

## Model Performance

| Metric | Score |
|--------|-------|
| **Accuracy** | 74.09% |
| **Precision (Diabetes)** | 71.95% |
| **Recall (Diabetes)** | 78.95% |
| **F1-Score (Diabetes)** | 75.00% |
| **ROC AUC** | 81.63% |
| **Cross-Validation ROC AUC** | 82.08% ± 0.57% |

### Model Type
- **Algorithm**: Random Forest Classifier
- **Features**: 24 engineered features (21 original + 3 derived)
- **Training Data**: 56,553 balanced samples
- **Test Data**: 14,139 samples
- **Cross-validation**: 5-fold CV used for model validation
- **Scikit-learn Version**: 1.7.1

### Dataset Balance
- **Original Dataset**: 253,680 samples with 6.18:1 imbalance ratio
- **After Balancing**: 70,692 samples (35,346 each class)
- **Balancing Method**: SMOTE (Synthetic Minority Oversampling Technique)

### Confusion Matrix Results
- **True Negatives**: 4,894 (correctly predicted non-diabetic)
- **False Positives**: 2,176 (incorrectly predicted diabetic)  
- **False Negatives**: 1,488 (missed diabetic cases)
- **True Positives**: 5,581 (correctly identified diabetic)

### Additional Metrics
- **Sensitivity (Diabetes Detection)**: 78.95%
- **Specificity (Non-Diabetes Detection)**: 69.22%
- **Positive Predictive Value**: 71.95%

## Installation

### Prerequisites
- Python 3.7+
- pip package manager

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/Vivektheprogrammer/ML_Diabetes_Risk_Prediction.git
   cd ML_Diabetes_Risk_Prediction
   ```

2. **Install dependencies**
   ```bash
   pip install streamlit pandas scikit-learn joblib numpy
   ```

3. **Run the application**
   ```bash
   streamlit run fullapp.py
   ```

4. **Open your browser** and navigate to `http://localhost:8501`

### Alternative Installation

You can also install specific versions:
```bash
pip install streamlit==1.28.0 pandas==2.0.3 scikit-learn==1.3.0 joblib==1.3.2 numpy==1.24.3
```

## Usage

### Web Application

1. **Launch the app**: Run `streamlit run fullapp.py`
2. **Input your details**: Fill in the health and lifestyle questionnaire
3. **Get prediction**: Click "Predict Diabetes Risk" for instant results
4. **Interpret results**: Review risk category and confidence score

### Input Parameters

The application requires the following inputs:

**Health Conditions:**
- High Blood Pressure (Yes/No)
- High Cholesterol (Yes/No)
- History of Stroke (Yes/No)
- Heart Disease or Attack (Yes/No)

**Lifestyle Factors:**
- Smoking History (Yes/No)
- Physical Activity (Yes/No)
- Fruit/Vegetable Consumption (Yes/No)
- Heavy Alcohol Consumption (Yes/No)

**Demographics & Physical:**
- Age Group (1-13 scale)
- Sex (Male/Female)
- BMI (Body Mass Index)
- Education Level (1-6 scale)
- Income Level (1-8 scale)

**Health Status:**
- General Health (1-5 scale)
- Mental Health Days (0-30)
- Physical Health Days (0-30)
- Difficulty Walking (Yes/No)

## Project Structure

```
Diabetes-Risk-Prediction/
│
├── fullapp.py                           # Main Streamlit application
├── app.py                              # Alternative simplified app
├── ML_diabetes.ipynb                   # Jupyter notebook with EDA and modeling
├── diabetes_binary_health_indicators_BRFSS2015.csv  # Dataset
│
├── Models/                             # Trained model files
│   ├── revised_diabetes_rf_model.joblib      # Final Random Forest model
│   ├── revised_model_features.joblib         # Feature names
│   ├── diabetes_rf_model_top.joblib         # Top features model
│   └── scaler_top.joblib                    # Feature scaler
│
├── README.md                           # Project documentation
└── requirements.txt                    # Python dependencies
```

## Model Details

### Algorithm Choice
- **Random Forest Classifier** was selected after comparing multiple algorithms
- Provides good balance between accuracy and interpretability
- Handles feature interactions well
- Robust to overfitting with 74.09% test accuracy

### Feature Engineering
- **Risk Factor Count**: Aggregated count of 7 major risk factors (0-7 range)
- **Age Group Code**: Categorical age groups for better pattern recognition
- **BMI Category Code**: Standard BMI classifications (0=Underweight, 1=Normal, 2=Overweight, 3=Obese)

### Model Training Process
1. **Data Preprocessing**: Handling missing values and encoding
2. **Class Balancing**: SMOTE technique to balance 6.18:1 imbalanced dataset
3. **Feature Engineering**: Creating 3 derived features from original 21
4. **Train-Test Split**: 80-20 split (56,553 training, 14,139 test samples)
5. **Model Training**: Random Forest with optimized parameters
6. **Cross-Validation**: 5-fold CV achieving 82.08% ± 0.57% ROC AUC
7. **Model Validation**: Comprehensive evaluation with confusion matrix analysis

## Deployment

### Streamlit Cloud

1. **Push to GitHub**: Ensure your code is in a GitHub repository
2. **Connect to Streamlit**: Visit [share.streamlit.io](https://share.streamlit.io)
3. **Deploy**: Select your repository and main file (`fullapp.py`)
4. **Configure**: Add any necessary secrets or environment variables

### Local Deployment

```bash
# Install Streamlit
pip install streamlit

# Run the application
streamlit run fullapp.py

# Access at http://localhost:8501
```

### Requirements File

Create a `requirements.txt` file:
```txt
streamlit>=1.28.0
pandas>=2.0.0
scikit-learn>=1.3.0
joblib>=1.3.0
numpy>=1.24.0
```

## Future Enhancements

- [ ] **Model Improvements**: Ensemble methods, deep learning approaches
- [ ] **Additional Features**: More health indicators, genetic factors
- [ ] **Data Updates**: Integration with newer BRFSS datasets
- [ ] **Mobile App**: React Native or Flutter implementation
- [ ] **API Development**: REST API for external integrations
- [ ] **Real-time Data**: Integration with wearable devices
- [ ] **Multi-language Support**: Internationalization capabilities

## Contributing

We welcome contributions! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Commit changes**: `git commit -m 'Add amazing feature'`
4. **Push to branch**: `git push origin feature/amazing-feature`
5. **Open a Pull Request**

### Development Guidelines
- Follow PEP 8 style guide
- Add tests for new features
- Update documentation as needed
- Ensure cross-platform compatibility

## Contact

**Vivek R** - [@Vivektheprogrammer](https://github.com/Vivektheprogrammer)

Project Link: [https://github.com/Vivektheprogrammer/ML_Diabetes_Risk_Prediction](https://github.com/Vivektheprogrammer/ML_Diabetes_Risk_Prediction)

---

**Star this repository if you found it helpful!** 

##  Disclaimer

This application is for educational and informational purposes only. The predictions made by this model should not be considered as medical advice. Always consult with healthcare professionals for proper medical diagnosis and treatment decisions.

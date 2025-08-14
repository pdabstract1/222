import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer

# Load the new model
model = joblib.load('RF.pkl')

# Load the test data from X_test.csv to create LIME explainer
X_test = pd.read_csv('X_test.csv')

# Define feature names from the new dataset
feature_names = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", 
    "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]

# Streamlit user interface
st.title("Heart Disease Predictor")

# Age: numerical input
age = st.number_input("Age:", min_value=0, max_value=120, value=41)

# Sex: categorical selection
sex = st.selectbox("Sex:", options=[0, 1], format_func=lambda x: "Male" if x == 1 else "Female")

# Chest Pain Type (cp): categorical selection
cp = st.selectbox("Chest Pain Type (CP):", options=[0, 1, 2, 3])

# Resting Blood Pressure (trestbps): numerical input
trestbps = st.number_input("Resting Blood Pressure (trestbps):", min_value=50, max_value=200, value=120)

# Cholesterol (chol): numerical input
chol = st.number_input("Cholesterol (chol):", min_value=100, max_value=600, value=157)

# Fasting Blood Sugar > 120 mg/dl (fbs): categorical selection
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (FBS):", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

# Resting Electrocardiographic Results (restecg): categorical selection
restecg = st.selectbox("Resting ECG (restecg):", options=[0, 1, 2])

# Maximum Heart Rate Achieved (thalach): numerical input
thalach = st.number_input("Maximum Heart Rate Achieved (thalach):", min_value=60, max_value=220, value=182)

# Exercise Induced Angina (exang): categorical selection
exang = st.selectbox("Exercise Induced Angina (exang):", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

# ST Depression Induced by Exercise (oldpeak): numerical input
oldpeak = st.number_input("ST Depression Induced by Exercise (oldpeak):", min_value=0.0, max_value=10.0, value=1.0)

# Slope of the Peak Exercise ST Segment (slope): categorical selection
slope = st.selectbox("Slope of the Peak Exercise ST Segment (slope):", options=[0, 1, 2])

# Number of Major Vessels Colored by Fluoroscopy (ca): categorical selection
ca = st.selectbox("Number of Major Vessels Colored by Fluoroscopy (ca):", options=[0, 1, 2, 3, 4])

# Thalassemia (thal): categorical selection
thal = st.selectbox("Thalassemia (thal):", options=[0, 1, 2, 3])

# Process inputs and make predictions
feature_values = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
features = np.array([feature_values])

if st.button("Predict"):
    # Predict class and probabilities
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]

    # Display prediction results
    st.write(f"**Predicted Class:** {predicted_class} (1: Disease, 0: No Disease)")
    st.write(f"**Prediction Probabilities:** {predicted_proba}")

    # Generate advice based on prediction results
    probability = predicted_proba[predicted_class] * 100

    if predicted_class == 1:
        advice = (
            f"According to our model, you have a high risk of heart disease. "
            f"The model predicts that your probability of having heart disease is {probability:.1f}%. "
            "It's advised to consult with your healthcare provider for further evaluation and possible intervention."
        )
    else:
        advice = (
            f"According to our model, you have a low risk of heart disease. "
            f"The model predicts that your probability of not having heart disease is {probability:.1f}%. "
            "However, maintaining a healthy lifestyle is important. Please continue regular check-ups with your healthcare provider."
        )

    st.write(advice)

    # SHAP Explanation
    st.subheader("SHAP Force Plot Explanation")
    explainer_shap = shap.TreeExplainer(model)
    shap_values = explainer_shap.shap_values(pd.DataFrame([feature_values], columns=feature_names))
    
    # Display the SHAP force plot for the predicted class
    if predicted_class == 1:
        shap.force_plot(explainer_shap.expected_value[1], shap_values[:,:,1], pd.DataFrame([feature_values], columns=feature_names), matplotlib=True)
    else:
        shap.force_plot(explainer_shap.expected_value[0], shap_values[:,:,0], pd.DataFrame([feature_values], columns=feature_names), matplotlib=True)

    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
    st.image("shap_force_plot.png", caption='SHAP Force Plot Explanation')

    # LIME Explanation
    st.subheader("LIME Explanation")
    lime_explainer = LimeTabularExplainer(
        training_data=X_test.values,
        feature_names=X_test.columns.tolist(),
        class_names=['Not sick', 'Sick'],  # Adjust class names to match your classification task
        mode='classification'
    )
    
    # Explain the instance
    lime_exp = lime_explainer.explain_instance(
        data_row=features.flatten(),
        predict_fn=model.predict_proba
    )

    # Display the LIME explanation without the feature value table
    lime_html = lime_exp.as_html(show_table=False)  # Disable feature value table
    st.components.v1.html(lime_html, height=800, scrolling=True)
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load saved model and scaler
model = joblib.load('random_forest_model.pkl')
scaler = joblib.load('scaler.pkl')

# App UI
st.set_page_config(page_title="Hospital Readmission Predictor", layout="centered")
st.title("Hospital Readmission Risk Predictor")

st.markdown("""
Use this app to predict if a patient is likely to be readmitted after discharge based on clinical details.
""")

# Inputs
age = st.slider("Age (in years)", 0, 100, 60)
time_in_hospital = st.slider("Time in hospital (days)", 1, 30, 4)
num_lab_procedures = st.slider("No. of lab procedures", 1, 100, 40)
num_procedures = st.slider("No. of procedures", 0, 10, 1)
num_medications = st.slider("No. of medications", 1, 100, 13)
number_outpatient = st.slider("Outpatient visits", 0, 20, 0)
number_emergency = st.slider("Emergency visits", 0, 10, 0)
number_inpatient = st.slider("Inpatient visits", 0, 20, 0)

# Data for prediction
input_data = np.array([[age, time_in_hospital, num_lab_procedures, num_procedures,
                        num_medications, number_outpatient, number_emergency, number_inpatient]])
scaled_input = scaler.transform(input_data)

# Predict
if st.button("Predict"):
    prediction = model.predict(scaled_input)[0]
    proba = model.predict_proba(scaled_input)[0][1] * 100

    if prediction == 1:
        st.error(f"High Risk of Readmission! ({proba:.2f}%)")
    else:
        st.success(f"Low Risk of Readmission ({100 - proba:.2f}%)")

    # Save prediction to CSV
    patient_data = {
        'Age': age,
        'Time_in_Hospital': time_in_hospital,
        'Lab_Procedures': num_lab_procedures,
        'Procedures': num_procedures,
        'Medications': num_medications,
        'Outpatient': number_outpatient,
        'Emergency': number_emergency,
        'Inpatient': number_inpatient,
        'Prediction': int(prediction),
        'Risk (%)': round(proba if prediction == 1 else 100 - proba, 2)
    }

    try:
        df_existing = pd.read_csv('predictions.csv')
        df_new = pd.DataFrame([patient_data])
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.to_csv('predictions.csv', index=False)
    except FileNotFoundError:
        pd.DataFrame([patient_data]).to_csv('predictions.csv', index=False)

    st.success("Prediction saved to predictions.csv")

    # Download Button
    with open("predictions.csv", "rb") as file:
        st.download_button("Download Predictions", file, "predictions.csv", "text/csv")

# Feature Importance Plot
st.subheader("Top Important Features")

feature_names = ['age', 'time_in_hospital', 'num_lab_procedures', 'num_procedures',
                 'num_medications', 'number_outpatient', 'number_emergency', 'number_inpatient']
importances = model.feature_importances_[:len(feature_names)]
feat_series = pd.Series(importances, index=feature_names).sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(x=feat_series.values, y=feat_series.index, ax=ax)
ax.set_title("Top Feature Importances")
st.pyplot(fig)
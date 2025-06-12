import joblib
import numpy as np
import pandas as pd

# Load model and scaler
model = joblib.load('random_forest_model.pkl')
scaler = joblib.load('scaler.pkl')

# Feature names used during training
feature_names = ['age', 'time_in_hospital', 'num_lab_procedures', 'num_procedures',
                 'num_medications', 'number_outpatient', 'number_emergency', 'number_inpatient']

# Sample input as a dictionary for one patient
input_dict = {
    'age': [5],  # e.g., encoded value for '70-80'
    'time_in_hospital': [6],
    'num_lab_procedures': [45],
    'num_procedures': [1],
    'num_medications': [13],
    'number_outpatient': [0],
    'number_emergency': [0],
    'number_inpatient': [0]
}

# Create DataFrame
input_df = pd.DataFrame(input_dict)

# Scale
scaled_input = scaler.transform(input_df)

# Predict
prediction = model.predict(scaled_input)

# Display
if prediction[0] == 1:
    print("Patient is likely to be readmitted (<30 days).")
else:
    print("Patient is NOT likely to be readmitted.")

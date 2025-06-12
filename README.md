# Hospital Readmission Risk Predictor

This project is a machine learning-based web app that predicts whether a diabetic patient is likely to be readmitted to the hospital within 30 days. It helps clinicians make informed decisions and offer personalized post-discharge plans based on a patient’s clinical profile.

## Project Highlights

- Predicts hospital readmission risk (<30 days, >30 days, or No readmission)
- Built using Random Forest, XGBoost, and Logistic Regression
- Final model accuracy: *83%*
- Trained on features like:
  - Number of lab procedures
  - Time in hospital
  - Number of previous visits (inpatient, outpatient, emergency)
  - Number of medications and procedures
  - Patient’s age group
- Integrated with a *Streamlit app* for real-time predictions

## Technologies Used

- *Python*
- *Pandas, **NumPy, **Scikit-learn, **XGBoost*
- *Matplotlib, **Seaborn* (for visualization)
- *Joblib* (for saving models)
- *Streamlit* (for web interface)

## Project Structure

hospital-readmission-predictor/
│
├── cleaned_diabetic_data.csv         # Input data
├── app.py                            # Streamlit app file
├── hospital_readmission_model.py     # Model training script
├── scaler.pkl                        # Saved StandardScaler
├── random_forest_model.pkl           # Trained Random Forest model
├── requirements.txt                  # Python dependencies
└── README.md                         # Project documentation
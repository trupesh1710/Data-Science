import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load("logistic_regression_model.pkl")

# Optional: Load scaler if used
# scaler = joblib.load("scaler.pkl")

st.title("Loan Default Prediction App")
st.write("Enter applicant details to predict loan default risk.")

# User inputs
income = st.number_input("Income", min_value=0)
age = st.number_input("Age", min_value=18, max_value=100)
experience = st.number_input("Years of Experience", min_value=0)
married_single = st.selectbox("Married (1) / Single (0)", [0, 1])
house_ownership = st.selectbox("House Ownership (1=Own, 0=Rent)", [0, 1])
car_ownership = st.selectbox("Car Ownership (1=Yes, 0=No)", [0, 1])
current_job_yrs = st.number_input("Current Job Years", min_value=0)

if st.button("Predict"):
    features = np.array([[income, age, experience, married_single, house_ownership, car_ownership, current_job_yrs]])
    
    # If you used a scaler:
    # features = scaler.transform(features)
    
    prediction = model.predict(features)
    prob = model.predict_proba(features)[0][1] * 100  # Probability of default
    
    if prediction[0] == 1:
        st.error(f"⚠ High Risk of Default ({prob:.2f}%)")
    else:
        st.success(f"✅ Low Risk of Default ({prob:.2f}%)")

import streamlit as st
import joblib
import pandas as pd

path = "logistic_regression_model.pkl"
model = joblib.load(path)

st.title("Loan Default Prediction App")
st.write("Enter applicant details to predict loan default risk.")

# User inputs (no default values)
income = st.text_input("Income")
age = st.text_input("Age")
experience = st.text_input("Years of Experience")
married_single = st.selectbox("Married (1) / Single (0)", [None, 0, 1], format_func=lambda x: "" if x is None else str(x))
house_ownership = st.selectbox("House Ownership (1=Own, 0=Rent)", [None, 0, 1], format_func=lambda x: "" if x is None else str(x))
car_ownership = st.selectbox("Car Ownership (1=Yes, 0=No)", [None, 0, 1], format_func=lambda x: "" if x is None else str(x))
current_job_yrs = st.text_input("Current Job Years")

if st.button("Predict"):
    # Validation: Ensure all inputs are provided
    if not income or not age or not experience or married_single is None or house_ownership is None or car_ownership is None or not current_job_yrs:
        st.error("⚠ Please fill in all the fields before predicting.")
    else:
        try:
            # Convert inputs to numeric
            features = pd.DataFrame([{
                'Income': float(income),
                'Age': float(age),
                'Experience': float(experience),
                'Married/Single': int(married_single),
                'House_Ownership': int(house_ownership),
                'Car_Ownership': int(car_ownership),
                'CURRENT_JOB_YRS': float(current_job_yrs)
            }])

            prediction = model.predict(features)
            prob = model.predict_proba(features)[0][1] * 100

            if prediction[0] == 1:
                st.error(f"⚠ High Risk of Default ({prob:.2f}%)")
            else:
                st.success(f"✅ Low Risk of Default ({prob:.2f}%)")
        except ValueError:
            st.error("⚠ Please enter valid numeric values in all fields.")

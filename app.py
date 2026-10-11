
import streamlit as st
import pandas as pd
import pickle

st.title("Customer Churn Prediction")
st.write("Predict whether a customer may leave.")

with open("churn_model.pkl", "rb") as f:
    model = pickle.load(f)

with st.form("customer_form"):
    tenure = st.number_input("Tenure (months)", 0, 100, 12)
    monthly = st.number_input("Monthly Charges", 0.0, 500.0, 70.0)
    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )
    internet = st.selectbox(
        "Internet Service", ["DSL", "Fiber optic", "No"]
    )
    submit = st.form_submit_button("Predict Churn")

if submit:
    # Fill the remaining columns with valid defaults
    sample = {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": tenure,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": internet,
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": contract,
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": monthly,
        "TotalCharges": tenure * monthly
    }

    X = pd.DataFrame([sample])
    result = model.predict(X)[0]
    probability = model.predict_proba(X)[0][1]

    st.write("Estimated churn probability:",
             f"{probability:.1%}")

    if result == 1:
        st.warning("The model predicts likely churn.")
    else:
        st.success("The model predicts likely retention.")

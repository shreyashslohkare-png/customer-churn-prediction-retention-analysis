import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os


# -------------------------------
# Page Configuration
# -------------------------------

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)


# -------------------------------
# Base Directory
# -------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "logistic_regression_model.pkl"
)

SCALER_PATH = os.path.join(
    BASE_DIR,
    "models",
    "scaler.pkl"
)

FEATURE_PATH = os.path.join(
    BASE_DIR,
    "models",
    "feature_columns.pkl"
)


# -------------------------------
# Load Model Files
# -------------------------------

@st.cache_resource
def load_files():

    model = joblib.load(MODEL_PATH)

    scaler = joblib.load(SCALER_PATH)

    feature_columns = joblib.load(FEATURE_PATH)

    return model, scaler, feature_columns


try:
    model, scaler, feature_columns = load_files()

except Exception as e:
    st.error("❌ Model files not found.")
    st.write("Check your GitHub folder structure.")
    st.write(e)
    st.stop()



# -------------------------------
# Title
# -------------------------------

st.title("📊 Customer Churn Prediction System")

st.write(
    """
    This application predicts whether a customer is likely to churn
    based on customer details.
    """
)


# -------------------------------
# User Input
# -------------------------------


st.sidebar.header("Customer Information")


gender = st.sidebar.selectbox(
    "Gender",
    ["Male", "Female"]
)


senior = st.sidebar.selectbox(
    "Senior Citizen",
    [0,1]
)


partner = st.sidebar.selectbox(
    "Partner",
    ["Yes","No"]
)


dependents = st.sidebar.selectbox(
    "Dependents",
    ["Yes","No"]
)


tenure = st.sidebar.number_input(
    "Tenure Months",
    min_value=0,
    max_value=100,
    value=12
)


phone = st.sidebar.selectbox(
    "Phone Service",
    ["Yes","No"]
)


multiple_lines = st.sidebar.selectbox(
    "Multiple Lines",
    ["Yes","No","No phone service"]
)


internet = st.sidebar.selectbox(
    "Internet Service",
    ["DSL","Fiber optic","No"]
)


online_security = st.sidebar.selectbox(
    "Online Security",
    ["Yes","No","No internet service"]
)


online_backup = st.sidebar.selectbox(
    "Online Backup",
    ["Yes","No","No internet service"]
)


device = st.sidebar.selectbox(
    "Device Protection",
    ["Yes","No","No internet service"]
)


tech_support = st.sidebar.selectbox(
    "Tech Support",
    ["Yes","No","No internet service"]
)


stream_tv = st.sidebar.selectbox(
    "Streaming TV",
    ["Yes","No","No internet service"]
)


stream_movies = st.sidebar.selectbox(
    "Streaming Movies",
    ["Yes","No","No internet service"]
)


contract = st.sidebar.selectbox(
    "Contract",
    [
        "Month-to-month",
        "One year",
        "Two year"
    ]
)


paperless = st.sidebar.selectbox(
    "Paperless Billing",
    ["Yes","No"]
)


payment = st.sidebar.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)


monthly = st.sidebar.number_input(
    "Monthly Charges",
    min_value=0.0,
    value=70.0
)


total = st.sidebar.number_input(
    "Total Charges",
    min_value=0.0,
    value=1000.0
)



# -------------------------------
# Prediction
# -------------------------------


if st.button("Predict Churn"):


    input_data = pd.DataFrame({

        "Gender":[gender],
        "Senior Citizen":[senior],
        "Partner":[partner],
        "Dependents":[dependents],
        "Tenure Months":[tenure],
        "Phone Service":[phone],
        "Multiple Lines":[multiple_lines],
        "Internet Service":[internet],
        "Online Security":[online_security],
        "Online Backup":[online_backup],
        "Device Protection":[device],
        "Tech Support":[tech_support],
        "Streaming TV":[stream_tv],
        "Streaming Movies":[stream_movies],
        "Contract":[contract],
        "Paperless Billing":[paperless],
        "Payment Method":[payment],
        "Monthly Charges":[monthly],
        "Total Charges":[total]

    })


    # One hot encoding

    input_encoded = pd.get_dummies(input_data)


    # Match training columns

    input_encoded = input_encoded.reindex(
        columns=feature_columns,
        fill_value=0
    )


    # Scaling

    input_scaled = scaler.transform(
        input_encoded
    )


    prediction = model.predict(
        input_scaled
    )


    probability = model.predict_proba(
        input_scaled
    )[0][1]



    st.subheader("Prediction Result")


    if prediction[0] == 1:

        st.error(
            f"⚠️ Customer likely to churn\n\nRisk Score: {probability:.2%}"
        )

    else:

        st.success(
            f"✅ Customer likely to stay\n\nChurn Probability: {probability:.2%}"
        )

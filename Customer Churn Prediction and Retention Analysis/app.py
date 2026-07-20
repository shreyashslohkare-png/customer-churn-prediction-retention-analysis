import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os


# ===============================
# PAGE CONFIGURATION
# ===============================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)


# ===============================
# FILE PATHS
# ===============================

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



# ===============================
# LOAD FILES
# ===============================

@st.cache_resource
def load_files():

    model = joblib.load(MODEL_PATH)

    scaler = joblib.load(SCALER_PATH)

    features = joblib.load(FEATURE_PATH)

    return model, scaler, features



try:

    model, scaler, feature_columns = load_files()

except Exception as e:

    st.error("Model loading failed")
    st.write(e)
    st.stop()



# ===============================
# TITLE
# ===============================

st.title("📊 Customer Churn Prediction System")

st.write(
"""
Predict whether a telecom customer is likely to churn.
"""
)



# ===============================
# INPUT SECTION
# ===============================

st.sidebar.header("Customer Details")



Gender = st.sidebar.selectbox(
    "Gender",
    ["Male","Female"]
)


SeniorCitizen = st.sidebar.selectbox(
    "Senior Citizen",
    [0,1]
)


Partner = st.sidebar.selectbox(
    "Partner",
    ["Yes","No"]
)


Dependents = st.sidebar.selectbox(
    "Dependents",
    ["Yes","No"]
)


Tenure = st.sidebar.number_input(
    "Tenure Months",
    min_value=0,
    value=12
)


PhoneService = st.sidebar.selectbox(
    "Phone Service",
    ["Yes","No"]
)


MultipleLines = st.sidebar.selectbox(
    "Multiple Lines",
    [
        "Yes",
        "No",
        "No phone service"
    ]
)


InternetService = st.sidebar.selectbox(
    "Internet Service",
    [
        "DSL",
        "Fiber optic",
        "No"
    ]
)


OnlineSecurity = st.sidebar.selectbox(
    "Online Security",
    [
        "Yes",
        "No",
        "No internet service"
    ]
)


OnlineBackup = st.sidebar.selectbox(
    "Online Backup",
    [
        "Yes",
        "No",
        "No internet service"
    ]
)


DeviceProtection = st.sidebar.selectbox(
    "Device Protection",
    [
        "Yes",
        "No",
        "No internet service"
    ]
)


TechSupport = st.sidebar.selectbox(
    "Tech Support",
    [
        "Yes",
        "No",
        "No internet service"
    ]
)


StreamingTV = st.sidebar.selectbox(
    "Streaming TV",
    [
        "Yes",
        "No",
        "No internet service"
    ]
)


StreamingMovies = st.sidebar.selectbox(
    "Streaming Movies",
    [
        "Yes",
        "No",
        "No internet service"
    ]
)


Contract = st.sidebar.selectbox(
    "Contract",
    [
        "Month-to-month",
        "One year",
        "Two year"
    ]
)


PaperlessBilling = st.sidebar.selectbox(
    "Paperless Billing",
    [
        "Yes",
        "No"
    ]
)


PaymentMethod = st.sidebar.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)


MonthlyCharges = st.sidebar.number_input(
    "Monthly Charges",
    min_value=0.0,
    value=70.0
)


TotalCharges = st.sidebar.number_input(
    "Total Charges",
    min_value=0.0,
    value=1000.0
)



# ===============================
# PREDICTION
# ===============================


if st.button("Predict Churn"):


    data = {

        "Gender":Gender,
        "Senior Citizen":SeniorCitizen,
        "Partner":Partner,
        "Dependents":Dependents,
        "Tenure Months":Tenure,
        "Phone Service":PhoneService,
        "Multiple Lines":MultipleLines,
        "Internet Service":InternetService,
        "Online Security":OnlineSecurity,
        "Online Backup":OnlineBackup,
        "Device Protection":DeviceProtection,
        "Tech Support":TechSupport,
        "Streaming TV":StreamingTV,
        "Streaming Movies":StreamingMovies,
        "Contract":Contract,
        "Paperless Billing":PaperlessBilling,
        "Payment Method":PaymentMethod,
        "Monthly Charges":MonthlyCharges,
        "Total Charges":TotalCharges

    }


    input_df = pd.DataFrame([data])


    # Encoding

    input_encoded = pd.get_dummies(input_df)



    # Match training columns

    input_encoded = input_encoded.reindex(
        columns=feature_columns,
        fill_value=0
    )



    # ===============================
    # SCALER HANDLING
    # ===============================

    if hasattr(scaler,"transform"):

        final_input = scaler.transform(
            input_encoded
        )

    else:

        final_input = input_encoded.values



    # Prediction

    prediction = model.predict(
        final_input
    )


    probability = None


    if hasattr(model,"predict_proba"):

        probability = model.predict_proba(
            final_input
        )[0][1]



    st.subheader("Prediction")



    if prediction[0] == 1:

        st.error("⚠️ Customer is likely to churn")

        if probability:

            st.write(
                f"Churn Probability: {probability:.2%}"
            )


    else:

        st.success(
            "✅ Customer is likely to stay"
        )

        if probability:

            st.write(
                f"Churn Probability: {probability:.2%}"
            )

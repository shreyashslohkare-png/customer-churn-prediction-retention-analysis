import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os


# ==============================
# PAGE CONFIG
# ==============================

st.set_page_config(
    page_title="Customer Churn Prediction & Retention Analysis",
    page_icon="📊",
    layout="wide"
)


# ==============================
# PATHS
# ==============================

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



# ==============================
# LOAD MODEL
# ==============================

@st.cache_resource
def load_files():

    model = joblib.load(MODEL_PATH)

    scaler = joblib.load(SCALER_PATH)

    features = joblib.load(FEATURE_PATH)

    return model, scaler, features



try:

    model, scaler, feature_columns = load_files()

except Exception as e:

    st.error("❌ Model loading failed")
    st.write(e)
    st.stop()



# ==============================
# TITLE
# ==============================

st.title("📊 Customer Churn Prediction & Retention Analysis")


st.write(
"""
This AI system predicts customer churn risk and provides
business recommendations to retain customers.
"""
)



# ==============================
# INPUTS
# ==============================


st.sidebar.header("Customer Information")


gender = st.sidebar.selectbox(
    "Gender",
    ["Male","Female"]
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
    value=12
)


phone = st.sidebar.selectbox(
    "Phone Service",
    ["Yes","No"]
)


multiple_lines = st.sidebar.selectbox(
    "Multiple Lines",
    [
        "Yes",
        "No",
        "No phone service"
    ]
)


internet = st.sidebar.selectbox(
    "Internet Service",
    [
        "DSL",
        "Fiber optic",
        "No"
    ]
)


security = st.sidebar.selectbox(
    "Online Security",
    [
        "Yes",
        "No",
        "No internet service"
    ]
)


backup = st.sidebar.selectbox(
    "Online Backup",
    [
        "Yes",
        "No",
        "No internet service"
    ]
)


device = st.sidebar.selectbox(
    "Device Protection",
    [
        "Yes",
        "No",
        "No internet service"
    ]
)


support = st.sidebar.selectbox(
    "Tech Support",
    [
        "Yes",
        "No",
        "No internet service"
    ]
)


tv = st.sidebar.selectbox(
    "Streaming TV",
    [
        "Yes",
        "No",
        "No internet service"
    ]
)


movies = st.sidebar.selectbox(
    "Streaming Movies",
    [
        "Yes",
        "No",
        "No internet service"
    ]
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
    [
        "Yes",
        "No"
    ]
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




# ==============================
# PREDICTION
# ==============================


if st.button("🔍 Predict Churn"):


    customer = pd.DataFrame({

        "Gender":[gender],
        "Senior Citizen":[senior],
        "Partner":[partner],
        "Dependents":[dependents],
        "Tenure Months":[tenure],
        "Phone Service":[phone],
        "Multiple Lines":[multiple_lines],
        "Internet Service":[internet],
        "Online Security":[security],
        "Online Backup":[backup],
        "Device Protection":[device],
        "Tech Support":[support],
        "Streaming TV":[tv],
        "Streaming Movies":[movies],
        "Contract":[contract],
        "Paperless Billing":[paperless],
        "Payment Method":[payment],
        "Monthly Charges":[monthly],
        "Total Charges":[total]

    })



    # Encoding

    customer_encoded = pd.get_dummies(customer)



    # Match training features

    customer_encoded = customer_encoded.reindex(
        columns=feature_columns,
        fill_value=0
    )



    # Scaling fix

    if hasattr(scaler,"transform"):

        final_data = scaler.transform(
            customer_encoded
        )

    else:

        final_data = customer_encoded.values



    prediction = model.predict(
        final_data
    )


    probability = 0


    if hasattr(model,"predict_proba"):

        probability = model.predict_proba(
            final_data
        )[0][1]



    # ==============================
    # RESULTS
    # ==============================


    st.subheader("Prediction Result")


    if prediction[0] == 1:


        st.error(
            "⚠️ Customer is likely to churn"
        )


        st.metric(
            "Churn Probability",
            f"{probability:.2%}"
        )


        st.subheader(
            "💡 Retention Recommendations"
        )


        recommendations=[]



        if contract=="Month-to-month":

            recommendations.append(
                "Offer yearly contract discounts to increase customer commitment."
            )


        if tenure < 12:

            recommendations.append(
                "Provide onboarding support and first-year loyalty benefits."
            )


        if monthly > 70:

            recommendations.append(
                "Provide personalized pricing offers because charges are high."
            )


        if support=="No":

            recommendations.append(
                "Offer free technical support packages."
            )


        if security=="No":

            recommendations.append(
                "Promote online security services."
            )


        if payment=="Electronic check":

            recommendations.append(
                "Encourage automatic payment methods with incentives."
            )


        if internet=="Fiber optic":

            recommendations.append(
                "Check service satisfaction and network issues."
            )



        if len(recommendations)==0:

            recommendations.append(
                "Provide loyalty rewards and customer engagement offers."
            )


        for item in recommendations:

            st.write(
                "🔹 "+item
            )



    else:


        st.success(
            "✅ Customer is likely to stay"
        )


        st.metric(
            "Churn Probability",
            f"{probability:.2%}"
        )


        st.subheader(
            "💡 Customer Engagement Suggestions"
        )


        st.write(
        """
        🔹 Continue loyalty programs  
        🔹 Offer premium upgrades  
        🔹 Maintain customer satisfaction  
        🔹 Encourage referrals
        """
        )

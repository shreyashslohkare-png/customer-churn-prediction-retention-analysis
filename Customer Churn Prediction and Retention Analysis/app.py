import streamlit as st
import pandas as pd
import joblib
import os


# ===============================
# PAGE CONFIG
# ===============================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)



# ===============================
# LOAD MODEL
# ===============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "logistic_regression_model.pkl"
)


@st.cache_resource
def load_model():

    model = joblib.load(MODEL_PATH)

    return model



try:

    model = load_model()

except Exception as e:

    st.error("Model loading failed")

    st.write(e)

    st.stop()



# ===============================
# TITLE
# ===============================


st.title(
    "📊 Customer Churn Prediction & Retention Analysis"
)


st.write(
"""
Predict customer churn probability and generate
personalized retention strategies.
"""
)



# ===============================
# INPUTS
# ===============================


st.sidebar.header(
    "Customer Details"
)


data = {}


data["Gender"] = st.sidebar.selectbox(
    "Gender",
    ["Male","Female"]
)


data["Senior Citizen"] = st.sidebar.selectbox(
    "Senior Citizen",
    [0,1]
)


data["Partner"] = st.sidebar.selectbox(
    "Partner",
    ["Yes","No"]
)


data["Dependents"] = st.sidebar.selectbox(
    "Dependents",
    ["Yes","No"]
)


data["Tenure Months"] = st.sidebar.number_input(
    "Tenure Months",
    0,
    100,
    12
)


data["Phone Service"] = st.sidebar.selectbox(
    "Phone Service",
    ["Yes","No"]
)


data["Multiple Lines"] = st.sidebar.selectbox(
    "Multiple Lines",
    [
        "Yes",
        "No",
        "No phone service"
    ]
)


data["Internet Service"] = st.sidebar.selectbox(
    "Internet Service",
    [
        "DSL",
        "Fiber optic",
        "No"
    ]
)


data["Online Security"] = st.sidebar.selectbox(
    "Online Security",
    [
        "Yes",
        "No",
        "No internet service"
    ]
)


data["Online Backup"] = st.sidebar.selectbox(
    "Online Backup",
    [
        "Yes",
        "No",
        "No internet service"
    ]
)


data["Device Protection"] = st.sidebar.selectbox(
    "Device Protection",
    [
        "Yes",
        "No",
        "No internet service"
    ]
)


data["Tech Support"] = st.sidebar.selectbox(
    "Tech Support",
    [
        "Yes",
        "No",
        "No internet service"
    ]
)


data["Streaming TV"] = st.sidebar.selectbox(
    "Streaming TV",
    [
        "Yes",
        "No",
        "No internet service"
    ]
)


data["Streaming Movies"] = st.sidebar.selectbox(
    "Streaming Movies",
    [
        "Yes",
        "No",
        "No internet service"
    ]
)


data["Contract"] = st.sidebar.selectbox(
    "Contract",
    [
        "Month-to-month",
        "One year",
        "Two year"
    ]
)


data["Paperless Billing"] = st.sidebar.selectbox(
    "Paperless Billing",
    [
        "Yes",
        "No"
    ]
)


data["Payment Method"] = st.sidebar.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)


data["Monthly Charges"] = st.sidebar.number_input(
    "Monthly Charges",
    value=70.0
)


data["Total Charges"] = st.sidebar.number_input(
    "Total Charges",
    value=1000.0
)



# ===============================
# PREDICTION
# ===============================


if st.button("Predict Churn"):


    input_df = pd.DataFrame([data])


    # same encoding style
    input_encoded = pd.get_dummies(
        input_df,
        drop_first=True,
        dtype=int
    )


    # use model expected columns directly
    expected_columns = model.feature_names_in_


    input_encoded = input_encoded.reindex(
        columns=expected_columns,
        fill_value=0
    )


    prediction = model.predict(
        input_encoded
    )


    probability = model.predict_proba(
        input_encoded
    )[0][1]



    st.divider()



    if prediction[0] == 1:


        st.error(
            "⚠️ Customer is likely to churn"
        )


        st.metric(
            "Churn Probability",
            f"{probability:.2%}"
        )



        st.subheader(
            "💊 Retention Prescription"
        )


        recommendations=[]


        if data["Contract"]=="Month-to-month":

            recommendations.append(
            "Offer annual contract upgrade with loyalty discount."
            )


        if data["Monthly Charges"]>80:

            recommendations.append(
            "Provide personalized pricing plan."
            )


        if data["Tech Support"]=="No":

            recommendations.append(
            "Provide technical support benefits."
            )


        if data["Tenure Months"]<12:

            recommendations.append(
            "Run customer onboarding campaign."
            )


        if len(recommendations)==0:

            recommendations.append(
            "Start customer engagement campaign."
            )


        for r in recommendations:

            st.info(r)



    else:


        st.success(
            "✅ Customer is unlikely to churn"
        )


        st.metric(
            "Churn Probability",
            f"{probability:.2%}"
        )


        st.write(
        """
        Retention Strategy:

        ✅ Maintain service quality

        ✅ Provide loyalty benefits

        ✅ Encourage upgrades
        """
        )



st.caption(
"Customer Churn Prediction & Retention Analysis"
)

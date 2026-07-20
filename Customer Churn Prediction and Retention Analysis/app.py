import streamlit as st
import pandas as pd
import joblib
import os


# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Customer Churn Prediction & Retention Analysis",
    page_icon="📊",
    layout="wide"
)


# ==========================================
# LOAD FILES
# ==========================================

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



@st.cache_resource
def load_model_files():

    model = joblib.load(MODEL_PATH)

    scaler = joblib.load(SCALER_PATH)

    features = joblib.load(FEATURE_PATH)

    return model, scaler, features



try:

    model, scaler, feature_columns = load_model_files()

except Exception as e:

    st.error("Model loading error")

    st.write(e)

    st.stop()



# ==========================================
# TITLE
# ==========================================


st.title(
    "📊 Customer Churn Prediction & Retention Analysis"
)


st.write(
"""
AI-based customer churn prediction system with
personalized retention recommendations.
"""
)



# ==========================================
# INPUTS
# ==========================================


st.sidebar.header(
    "Customer Information"
)



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
    0,
    100,
    12
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


stream_tv = st.sidebar.selectbox(
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
    0.0,
    200.0,
    70.0
)


total = st.sidebar.number_input(
    "Total Charges",
    0.0,
    10000.0,
    1000.0
)



# ==========================================
# PREDICTION
# ==========================================


if st.button(
    "🔍 Analyze Customer"
):


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

        "Streaming TV":[stream_tv],

        "Streaming Movies":[movies],

        "Contract":[contract],

        "Paperless Billing":[paperless],

        "Payment Method":[payment],

        "Monthly Charges":[monthly],

        "Total Charges":[total]

    })



    encoded = pd.get_dummies(customer)



    encoded = encoded.reindex(
        columns=feature_columns,
        fill_value=0
    )



    if hasattr(
        scaler,
        "transform"
    ):

        final_input = scaler.transform(
            encoded
        )

    else:

        final_input = encoded.values



    prediction = model.predict(
        final_input
    )


    probability = model.predict_proba(
        final_input
    )[0][1]



    st.divider()



    # ==========================================
    # RESULT
    # ==========================================


    if prediction[0] == 1:


        st.error(
            "⚠️ Customer is likely to churn"
        )


        st.metric(
            "Churn Probability",
            f"{probability:.2%}"
        )



        st.subheader(
            "🔎 Customer Risk Analysis"
        )



        risks=[]



        if contract=="Month-to-month":

            risks.append(
                (
                "Contract Risk",
                "Customer has flexible contract with high switching possibility."
                )
            )



        if monthly>80:

            risks.append(
                (
                "Pricing Risk",
                "Monthly charges are high compared with average customers."
                )
            )



        if tenure<12:

            risks.append(
                (
                "Relationship Risk",
                "Customer relationship is still new."
                )
            )



        if support=="No":

            risks.append(
                (
                "Service Risk",
                "Customer does not have technical support."
                )
            )



        if payment=="Electronic check":

            risks.append(
                (
                "Payment Risk",
                "Payment method is associated with higher churn."
                )
            )



        if len(risks)==0:

            risks.append(
                (
                "General Risk",
                "No specific risk factor detected."
                )
            )



        for title,desc in risks:

            st.warning(
                f"**{title}**\n\n{desc}"
            )



        st.subheader(
            "💊 Personalized Retention Prescription"
        )



        for title,desc in risks:


            if title=="Contract Risk":

                st.info(
                "Offer yearly contract upgrade with loyalty discount."
                )


            elif title=="Pricing Risk":

                st.info(
                "Provide customized pricing plan or promotional offer."
                )


            elif title=="Service Risk":

                st.info(
                "Provide free technical support and service assistance."
                )


            elif title=="Relationship Risk":

                st.info(
                "Start onboarding campaign and provide welcome benefits."
                )


            elif title=="Payment Risk":

                st.info(
                "Encourage automatic payment with incentives."
                )



        st.subheader(
            "🚨 Customer Priority"
        )


        if probability >=0.75:

            st.error(
            "HIGH PRIORITY: Contact customer immediately."
            )


        else:

            st.warning(
            "MEDIUM PRIORITY: Start targeted retention campaign."
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
            "💡 Customer Engagement Strategy"
        )


        st.write(
        """
        ✅ Continue loyalty benefits

        ✅ Offer premium services

        ✅ Maintain service quality

        ✅ Collect customer feedback
        """
        )



st.divider()

st.caption(
"Machine Learning | Customer Analytics | Retention Strategy"
)

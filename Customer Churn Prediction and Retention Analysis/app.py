import streamlit as st
import pandas as pd
import joblib
import os


# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="Customer Churn Prediction & Retention Analysis",
    page_icon="📊",
    layout="wide"
)


# =====================================
# LOAD MODEL
# =====================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "logistic_regression_model.pkl"
)


@st.cache_resource
def load_model():

    return joblib.load(MODEL_PATH)



try:

    model = load_model()

except Exception as e:

    st.error("Model loading failed")

    st.write(e)

    st.stop()



# =====================================
# HEADER
# =====================================

st.title(
    "📊 Customer Churn Prediction & Retention Analysis"
)


st.write(
"""
An AI-based customer analytics system that predicts churn risk
and generates personalized retention strategies.
"""
)



# =====================================
# CUSTOMER INPUT
# =====================================


st.sidebar.header(
    "Customer Information"
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



# =====================================
# PREDICTION
# =====================================


if st.button("🔍 Analyze Customer"):


    input_df = pd.DataFrame(
        [data]
    )


    # SAME PREPROCESSING AS TRAINING

    input_encoded = pd.get_dummies(
        input_df,
        drop_first=True,
        dtype=int
    )


    # MATCH MODEL FEATURES

    expected_features = model.feature_names_in_


    input_encoded = input_encoded.reindex(
        columns=expected_features,
        fill_value=0
    )



    prediction = model.predict(
        input_encoded
    )


    probability = model.predict_proba(
        input_encoded
    )[0][1]



    st.divider()



    # =====================================
    # CHURN CUSTOMER
    # =====================================


    if prediction[0] == 1:


        st.error(
            "⚠️ CUSTOMER IS LIKELY TO CHURN"
        )


        st.metric(
            "Churn Probability",
            f"{probability:.2%}"
        )



        # -------------------------------
        # EXPLANATION
        # -------------------------------

        st.subheader(
            "🧠 Why did the model predict churn?"
        )


        risks=[]



        if data["Contract"]=="Month-to-month":

            risks.append(
            """
            **Contract Risk**

            Month-to-month customers have higher switching probability
            because they have low commitment.
            """
            )


        if data["Monthly Charges"]>80:

            risks.append(
            """
            **Pricing Risk**

            Customer has high monthly charges which may create dissatisfaction.
            """
            )



        if data["Tenure Months"]<12:

            risks.append(
            """
            **Customer Loyalty Risk**

            Customer is new and has not developed strong loyalty.
            """
            )



        if data["Tech Support"]=="No":

            risks.append(
            """
            **Service Risk**

            Customer does not have technical support,
            increasing service frustration.
            """
            )


        if data["Payment Method"]=="Electronic check":

            risks.append(
            """
            **Payment Risk**

            Electronic check payment method is associated
            with higher churn probability.
            """
            )



        if len(risks)==0:

            risks.append(
            """
            Model detected a pattern similar to previously churned customers.
            """
            )



        for r in risks:

            st.warning(r)



        # -------------------------------
        # RETENTION PRESCRIPTION
        # -------------------------------


        st.subheader(
            "💊 Personalized Retention Prescription"
        )


        actions=[]



        if data["Contract"]=="Month-to-month":

            actions.append(
            "Offer yearly contract upgrade with loyalty discount."
            )



        if data["Monthly Charges"]>80:

            actions.append(
            "Provide customized pricing package."
            )



        if data["Tech Support"]=="No":

            actions.append(
            "Provide free technical support trial."
            )



        if data["Tenure Months"]<12:

            actions.append(
            "Start onboarding and customer engagement campaign."
            )



        if data["Payment Method"]=="Electronic check":

            actions.append(
            "Encourage automatic payment methods with incentives."
            )



        actions.append(
        "Add customer to loyalty program and satisfaction monitoring."
        )



        for action in actions:

            st.info(action)



        # -------------------------------
        # PRIORITY
        # -------------------------------


        st.subheader(
            "🚨 Retention Priority"
        )


        if probability >=0.75:

            st.error(
            """
            HIGH PRIORITY CUSTOMER

            Contact customer immediately with a personalized retention offer.
            """
            )


        elif probability >=0.5:

            st.warning(
            """
            MEDIUM PRIORITY CUSTOMER

            Start targeted retention campaign.
            """
            )


        else:

            st.success(
            """
            LOW PRIORITY CUSTOMER

            Continue normal engagement.
            """
            )



    # =====================================
    # SAFE CUSTOMER
    # =====================================

    else:


        st.success(
            "✅ CUSTOMER IS UNLIKELY TO CHURN"
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
        ✅ Maintain service quality

        ✅ Provide loyalty rewards

        ✅ Encourage upgrades

        ✅ Collect feedback regularly
        """
        )



st.divider()


st.caption(
"Customer Churn Prediction & Retention Analysis | Machine Learning Project"
)

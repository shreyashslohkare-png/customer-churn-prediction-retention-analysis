import streamlit as st
import pandas as pd
import joblib


# ---------------------------------------------------
# Page Configuration
# ---------------------------------------------------

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)



# ---------------------------------------------------
# Load Model
# ---------------------------------------------------

@st.cache_resource
def load_model():

    model = joblib.load(
        "models/logistic_regression_model.pkl"
    )

    return model



model = load_model()



# ---------------------------------------------------
# Title
# ---------------------------------------------------

st.title("📊 Customer Churn Prediction System")

st.write(
    "Predict whether a telecom customer is likely to churn."
)

st.divider()



# ---------------------------------------------------
# Input Section
# ---------------------------------------------------

st.header("📝 Customer Information")


col1, col2 = st.columns(2)



with col1:


    gender = st.selectbox(
        "Gender",
        ["Female","Male"]
    )


    senior = st.selectbox(
        "Senior Citizen",
        ["No","Yes"]
    )


    partner = st.selectbox(
        "Partner",
        ["No","Yes"]
    )


    dependents = st.selectbox(
        "Dependents",
        ["No","Yes"]
    )


    tenure = st.number_input(
        "Tenure Months",
        min_value=0,
        max_value=72,
        value=12
    )


    monthly = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        max_value=200.0,
        value=70.0
    )


    online_security = st.selectbox(
        "Online Security",
        ["No","Yes","No internet service"]
    )


    online_backup = st.selectbox(
        "Online Backup",
        ["No","Yes","No internet service"]
    )


    device_protection = st.selectbox(
        "Device Protection",
        ["No","Yes","No internet service"]
    )



with col2:


    phone = st.selectbox(
        "Phone Service",
        ["No","Yes"]
    )


    multiple = st.selectbox(
        "Multiple Lines",
        [
            "No",
            "Yes",
            "No phone service"
        ]
    )


    internet = st.selectbox(
        "Internet Service",
        [
            "DSL",
            "Fiber optic",
            "No"
        ]
    )


    tech_support = st.selectbox(
        "Tech Support",
        [
            "No",
            "Yes",
            "No internet service"
        ]
    )


    streaming_tv = st.selectbox(
        "Streaming TV",
        [
            "No",
            "Yes",
            "No internet service"
        ]
    )


    streaming_movies = st.selectbox(
        "Streaming Movies",
        [
            "No",
            "Yes",
            "No internet service"
        ]
    )


    contract = st.selectbox(
        "Contract",
        [
            "Month-to-month",
            "One year",
            "Two year"
        ]
    )


    paperless = st.selectbox(
        "Paperless Billing",
        [
            "No",
            "Yes"
        ]
    )


    payment = st.selectbox(
        "Payment Method",
        [
            "Bank transfer (automatic)",
            "Credit card (automatic)",
            "Electronic check",
            "Mailed check"
        ]
    )



total = round(
    tenure * monthly,
    2
)



st.metric(
    "Total Charges",
    f"${total}"
)



st.divider()



predict = st.button(
    "🔍 Predict Churn",
    use_container_width=True
)



# ---------------------------------------------------
# Prediction
# ---------------------------------------------------

if predict:


    data = pd.DataFrame({

        "Gender":[gender],

        "Senior Citizen":[senior],

        "Partner":[partner],

        "Dependents":[dependents],

        "Tenure Months":[tenure],

        "Phone Service":[phone],

        "Multiple Lines":[multiple],

        "Internet Service":[internet],

        "Online Security":[online_security],

        "Online Backup":[online_backup],

        "Device Protection":[device_protection],

        "Tech Support":[tech_support],

        "Streaming TV":[streaming_tv],

        "Streaming Movies":[streaming_movies],

        "Contract":[contract],

        "Paperless Billing":[paperless],

        "Payment Method":[payment],

        "Monthly Charges":[monthly],

        "Total Charges":[total]

    })



    # -------------------------------
    # One Hot Encoding
    # -------------------------------

    data = pd.get_dummies(data)



    # -------------------------------
    # Match EXACT MODEL FEATURES
    # -------------------------------

    model_features = model.feature_names_in_


    data = data.reindex(
        columns=model_features,
        fill_value=0
    )



    # -------------------------------
    # Debug Check
    # -------------------------------

    if list(data.columns) != list(model_features):

        st.error(
            "Feature mismatch detected"
        )

        st.write(
            "Expected:"
        )

        st.write(model_features)


        st.write(
            "Received:"
        )

        st.write(data.columns)


        st.stop()



    # -------------------------------
    # Prediction
    # -------------------------------

    prediction = model.predict(
        data
    )[0]


    probability = model.predict_proba(
        data
    )[0][1]



    st.divider()


    st.header(
        "📌 Result"
    )



    if prediction == 1:

        st.error(
            "⚠ Customer is likely to CHURN"
        )


    else:

        st.success(
            "✅ Customer is likely to STAY"
        )



    st.subheader(
        "Churn Probability"
    )


    st.progress(
        float(probability)
    )


    st.write(
        f"{probability:.2%}"
    )



    if probability >= 0.8:

        st.error(
            "🔴 Very High Risk"
        )

    elif probability >= 0.6:

        st.warning(
            "🟠 High Risk"
        )

    elif probability >= 0.4:

        st.info(
            "🟡 Medium Risk"
        )

    else:

        st.success(
            "🟢 Low Risk"
        )



    st.divider()



    st.subheader(
        "💡 Retention Suggestions"
    )


    if prediction == 1:

        st.write(
            """
            - Offer personalized discounts
            - Encourage yearly contracts
            - Provide loyalty rewards
            - Improve customer support
            """
        )

    else:

        st.write(
            """
            - Maintain engagement
            - Offer premium services
            - Build customer loyalty
            """
        )
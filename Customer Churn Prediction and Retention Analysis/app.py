import streamlit as st
import pandas as pd
import joblib
import os


# =====================================
# PAGE CONFIGURATION
# =====================================

st.set_page_config(
    page_title="Customer Churn Prediction & Retention Analysis",
    page_icon="📊",
    layout="wide"
)


# =====================================
# LOAD MODEL FILES
# =====================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "logistic_regression_model.pkl"
)

FEATURE_PATH = os.path.join(
    BASE_DIR,
    "models",
    "feature_columns.pkl"
)



@st.cache_resource
def load_files():

    model = joblib.load(MODEL_PATH)

    feature_columns = joblib.load(FEATURE_PATH)

    return model, feature_columns



try:

    model, feature_columns = load_files()

except Exception as e:

    st.error("Model files not found")

    st.write(e)

    st.stop()



# =====================================
# TITLE
# =====================================

st.title(
    "📊 Customer Churn Prediction & Retention Analysis"
)

st.write(
"""
Machine Learning system that predicts customer churn risk
and generates personalized retention strategies.
"""
)



# =====================================
# CUSTOMER INPUTS
# =====================================


st.sidebar.header(
    "Customer Information"
)


Gender = st.sidebar.selectbox(
    "Gender",
    ["Male","Female"]
)


Senior_Citizen = st.sidebar.selectbox(
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


Tenure_Months = st.sidebar.number_input(
    "Tenure Months",
    min_value=0,
    max_value=100,
    value=12
)


Phone_Service = st.sidebar.selectbox(
    "Phone Service",
    ["Yes","No"]
)


Multiple_Lines = st.sidebar.selectbox(
    "Multiple Lines",
    [
        "Yes",
        "No",
        "No phone service"
    ]
)


Internet_Service = st.sidebar.selectbox(
    "Internet Service",
    [
        "DSL",
        "Fiber optic",
        "No"
    ]
)


Online_Security = st.sidebar.selectbox(
    "Online Security",
    [
        "Yes",
        "No",
        "No internet service"
    ]
)


Online_Backup = st.sidebar.selectbox(
    "Online Backup",
    [
        "Yes",
        "No",
        "No internet service"
    ]
)


Device_Protection = st.sidebar.selectbox(
    "Device Protection",
    [
        "Yes",
        "No",
        "No internet service"
    ]
)


Tech_Support = st.sidebar.selectbox(
    "Tech Support",
    [
        "Yes",
        "No",
        "No internet service"
    ]
)


Streaming_TV = st.sidebar.selectbox(
    "Streaming TV",
    [
        "Yes",
        "No",
        "No internet service"
    ]
)


Streaming_Movies = st.sidebar.selectbox(
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


Paperless_Billing = st.sidebar.selectbox(
    "Paperless Billing",
    [
        "Yes",
        "No"
    ]
)


Payment_Method = st.sidebar.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)


Monthly_Charges = st.sidebar.number_input(
    "Monthly Charges",
    value=70.0
)


Total_Charges = st.sidebar.number_input(
    "Total Charges",
    value=1000.0
)



# =====================================
# PREDICTION
# =====================================


if st.button("🔍 Predict Churn"):


    input_data = pd.DataFrame({

        "Gender":[Gender],

        "Senior Citizen":[Senior_Citizen],

        "Partner":[Partner],

        "Dependents":[Dependents],

        "Tenure Months":[Tenure_Months],

        "Phone Service":[Phone_Service],

        "Multiple Lines":[Multiple_Lines],

        "Internet Service":[Internet_Service],

        "Online Security":[Online_Security],

        "Online Backup":[Online_Backup],

        "Device Protection":[Device_Protection],

        "Tech Support":[Tech_Support],

        "Streaming TV":[Streaming_TV],

        "Streaming Movies":[Streaming_Movies],

        "Contract":[Contract],

        "Paperless Billing":[Paperless_Billing],

        "Payment Method":[Payment_Method],

        "Monthly Charges":[Monthly_Charges],

        "Total Charges":[Total_Charges]

    })



    # Same preprocessing as training

    categorical_columns = input_data.select_dtypes(
        include="object"
    ).columns



    input_encoded = pd.get_dummies(
        input_data,
        columns=categorical_columns,
        drop_first=True,
        dtype=int
    )



    # Match training columns

    input_encoded = input_encoded.reindex(
        columns=feature_columns,
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
    # RESULT
    # =====================================


    if prediction[0]==1:


        st.error(
            "⚠️ Customer is likely to churn"
        )


        st.metric(
            "Churn Probability",
            f"{probability:.2%}"
        )



        st.subheader(
            "🔎 Risk Analysis"
        )


        risks=[]



        if Contract=="Month-to-month":

            risks.append(
            "Customer has month-to-month contract with high switching possibility."
            )


        if Monthly_Charges>80:

            risks.append(
            "High monthly charges may create price dissatisfaction."
            )


        if Tenure_Months<12:

            risks.append(
            "Customer relationship is new and loyalty is weak."
            )


        if Tech_Support=="No":

            risks.append(
            "No technical support may increase frustration."
            )


        if Payment_Method=="Electronic check":

            risks.append(
            "Electronic check users show higher churn tendency."
            )



        for r in risks:

            st.warning(r)



        st.subheader(
            "💊 Retention Prescription"
        )


        for r in risks:


            if "contract" in r.lower():

                st.info(
                "Offer yearly contract upgrade with loyalty benefits."
                )


            elif "charges" in r.lower():

                st.info(
                "Provide personalized pricing plan or discount."
                )


            elif "support" in r.lower():

                st.info(
                "Offer free technical support package."
                )


            elif "relationship" in r.lower():

                st.info(
                "Start customer onboarding and engagement campaign."
                )


            elif "payment" in r.lower():

                st.info(
                "Encourage automatic payment methods."
                )



        if probability>=0.75:

            st.error(
            "🚨 HIGH PRIORITY: Contact customer immediately."
            )

        else:

            st.warning(
            "⚠️ MEDIUM PRIORITY: Add to retention campaign."
            )



    else:


        st.success(
            "✅ Customer is unlikely to churn"
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

        ✅ Offer loyalty rewards

        ✅ Encourage premium services

        ✅ Collect customer feedback
        """
        )



st.divider()

st.caption(
"Customer Churn Prediction & Retention Analysis | Machine Learning Project"
)

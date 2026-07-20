import streamlit as st
import pandas as pd
import joblib
import os


# ===============================
# PAGE SETTINGS
# ===============================

st.set_page_config(
    page_title="Customer Churn Prediction & Retention Analysis",
    page_icon="📊",
    layout="wide"
)


# ===============================
# PATHS
# ===============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR,"models","logistic_regression_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR,"models","scaler.pkl")
FEATURE_PATH = os.path.join(BASE_DIR,"models","feature_columns.pkl")


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

    st.error("Model files loading failed")
    st.write(e)
    st.stop()



# ===============================
# TITLE
# ===============================

st.title("📊 Customer Churn Prediction & Retention Analysis")

st.write(
"""
AI powered system that predicts customer churn risk
and generates personalized retention strategies.
"""
)



# ===============================
# INPUTS
# ===============================

st.sidebar.header("Customer Details")


gender = st.sidebar.selectbox("Gender",["Male","Female"])

senior = st.sidebar.selectbox("Senior Citizen",[0,1])

partner = st.sidebar.selectbox("Partner",["Yes","No"])

dependents = st.sidebar.selectbox("Dependents",["Yes","No"])

tenure = st.sidebar.number_input(
    "Tenure Months",
    min_value=0,
    value=12
)

phone = st.sidebar.selectbox(
    "Phone Service",
    ["Yes","No"]
)

multiple = st.sidebar.selectbox(
    "Multiple Lines",
    ["Yes","No","No phone service"]
)

internet = st.sidebar.selectbox(
    "Internet Service",
    ["DSL","Fiber optic","No"]
)

security = st.sidebar.selectbox(
    "Online Security",
    ["Yes","No","No internet service"]
)

backup = st.sidebar.selectbox(
    "Online Backup",
    ["Yes","No","No internet service"]
)

device = st.sidebar.selectbox(
    "Device Protection",
    ["Yes","No","No internet service"]
)

support = st.sidebar.selectbox(
    "Tech Support",
    ["Yes","No","No internet service"]
)

tv = st.sidebar.selectbox(
    "Streaming TV",
    ["Yes","No","No internet service"]
)

movies = st.sidebar.selectbox(
    "Streaming Movies",
    ["Yes","No","No internet service"]
)

contract = st.sidebar.selectbox(
    "Contract",
    ["Month-to-month","One year","Two year"]
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
    value=70.0
)

total = st.sidebar.number_input(
    "Total Charges",
    value=1000.0
)



# ===============================
# PREDICTION
# ===============================

if st.button("🔍 Predict Customer Risk"):


    customer = pd.DataFrame({

        "Gender":[gender],
        "Senior Citizen":[senior],
        "Partner":[partner],
        "Dependents":[dependents],
        "Tenure Months":[tenure],
        "Phone Service":[phone],
        "Multiple Lines":[multiple],
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


    encoded = pd.get_dummies(customer)


    encoded = encoded.reindex(
        columns=feature_columns,
        fill_value=0
    )


    # scaler handling

    if hasattr(scaler,"transform"):

        final_input = scaler.transform(encoded)

    else:

        final_input = encoded.values



    prediction = model.predict(final_input)


    probability = 0

    if hasattr(model,"predict_proba"):

        probability = model.predict_proba(final_input)[0][1]



    st.divider()


    # ===============================
    # RESULT
    # ===============================


    if prediction[0]==1:


        st.error("⚠️ HIGH CHURN RISK CUSTOMER")


        st.metric(
            "Churn Probability",
            f"{probability:.2%}"
        )


        st.subheader("💊 Customer Retention Prescription")


        st.markdown("### 🔎 Risk Analysis")


        risks=[]


        if contract=="Month-to-month":

            risks.append(
            "Customer uses month-to-month contract, showing low commitment."
            )


        if tenure<12:

            risks.append(
            "Customer is new and loyalty is not established."
            )


        if monthly>70:

            risks.append(
            "High monthly charges may create pricing dissatisfaction."
            )


        if support=="No":

            risks.append(
            "No technical support increases service frustration."
            )


        if security=="No":

            risks.append(
            "Missing security services reduces customer engagement."
            )


        if payment=="Electronic check":

            risks.append(
            "Electronic payment behaviour indicates higher churn tendency."
            )


        if len(risks)==0:

            risks.append(
            "No major risk factor detected."
            )


        for r in risks:

            st.warning(r)



        st.markdown("### 🎯 Recommended Retention Actions")


        actions=[]


        if contract=="Month-to-month":

            actions.append(
            "Offer yearly contract upgrade with loyalty discount."
            )


        if monthly>70:

            actions.append(
            "Provide personalized pricing package."
            )


        if support=="No":

            actions.append(
            "Provide free technical support trial."
            )


        if tenure<12:

            actions.append(
            "Create onboarding and welcome reward program."
            )


        if payment=="Electronic check":

            actions.append(
            "Encourage automatic payment with incentives."
            )


        actions.append(
        "Add customer to loyalty program and satisfaction monitoring."
        )


        for a in actions:

            st.info(a)



        st.markdown("### 🚨 Priority Level")


        if probability>=0.75:

            st.error(
            "HIGH PRIORITY: Contact customer immediately with retention offer."
            )

        elif probability>=0.50:

            st.warning(
            "MEDIUM PRIORITY: Start engagement campaign."
            )

        else:

            st.success(
            "LOW PRIORITY: Continue normal relationship management."
            )



    else:


        st.success("✅ LOW CHURN RISK CUSTOMER")


        st.metric(
            "Churn Probability",
            f"{probability:.2%}"
        )


        st.subheader("💡 Customer Engagement Plan")


        st.write(
        """
        ✅ Continue loyalty benefits

        ✅ Maintain service quality

        ✅ Encourage upgrades

        ✅ Collect feedback regularly
        """
        )



st.divider()

st.caption(
"Customer Churn Prediction & Retention Analysis | Machine Learning + Business Recommendations"
)

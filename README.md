

# 📊 Customer Churn Prediction and Retention Analysis

## 🚀 Project Overview

Customer churn is one of the biggest challenges faced by subscription-based businesses. Losing customers directly impacts revenue, growth, and long-term business sustainability.

This project focuses on building a **Machine Learning-based Customer Churn Prediction System** that predicts whether a telecom customer is likely to leave the service or continue using it.

The project includes:

- Data cleaning and preprocessing
- Exploratory Data Analysis (EDA)
- Feature engineering
- Machine Learning model development
- Model evaluation
- Customer churn prediction application
- Streamlit deployment interface
- Retention strategy recommendations


---

# 🎯 Problem Statement

Telecom companies collect large amounts of customer data but often struggle to identify customers who are likely to churn.

The objective of this project is to:

- Predict customers who are at risk of leaving
- Understand major factors affecting churn
- Help businesses take proactive retention actions


---

# 🧠 Machine Learning Approach

Data Collection
|
↓
Data Cleaning
|
↓
Exploratory Data Analysis
|
↓
Feature Engineering
|
↓
Model Training
|
↓
Model Evaluation
|
↓
Deployment using Streamlit

The project follows a complete ML workflow:


---

# 📂 Dataset Information

The dataset used is a telecom customer dataset containing customer demographics, service information, and billing details.

### Dataset Features

### Customer Information

- Customer ID
- Gender
- Senior Citizen
- Partner
- Dependents


### Service Information

- Phone Service
- Multiple Lines
- Internet Service
- Online Security
- Online Backup
- Device Protection
- Tech Support
- Streaming TV
- Streaming Movies


### Account Information

- Tenure Months
- Contract Type
- Paperless Billing
- Payment Method


### Billing Information

- Monthly Charges
- Total Charges


### Target Variable

Churn Value

0 → Customer stays
1 → Customer churns



---

# 🔍 Exploratory Data Analysis

EDA was performed to understand customer behavior and identify churn patterns.

Important business questions analyzed:

### 1. Which contract type has the highest churn?

Finding:

- Month-to-month customers show significantly higher churn.
- Long-term contracts have better customer retention.


### 2. Does tenure affect churn?

Finding:

- New customers have a higher probability of leaving.
- Customers with longer relationships are more stable.


### 3. Does payment method influence churn?

Finding:

- Electronic check users show higher churn tendency.


### 4. Which services impact customer retention?

Finding:

- Customers without additional support services have higher churn risk.


---

# 🤖 Machine Learning Model

## Model Used

### Logistic Regression

Reason for selection:

- Works well for binary classification problems
- Provides probability scores
- Easy interpretation for business decisions


---

# ⚙️ Data Preprocessing

The following preprocessing steps were applied:

- Removed unnecessary columns
- Converted numerical features
- Handled categorical variables
- Applied one-hot encoding
- Prepared features for machine learning


---

# 📊 Model Output

The model predicts:

### Prediction

0 → Customer will stay

1 → Customer will churn



### Probability Score

The application also provides:

- Churn probability percentage
- Risk category


Risk Levels:



### Probability Score

The application also provides:

- Churn probability percentage
- Risk category


Risk Levels:

80%+ → Very High Risk

60-80% → High Risk

40-60% → Medium Risk

Below 40% → Low Risk



---

# 🖥️ Streamlit Application

A web application was created using Streamlit.

The application allows users to:

- Enter customer details
- Predict churn probability
- View risk level
- Get retention suggestions


---

# 📸 Application Features

## Customer Input

Users can provide:

- Customer demographics
- Services used
- Contract details
- Billing information


## Prediction Result

The application displays:

✅ Customer likely to stay

or

⚠ Customer likely to churn


## Retention Suggestions

For high-risk customers:

- Offer personalized discounts
- Encourage yearly contracts
- Provide loyalty rewards
- Improve customer support


---

# 🛠️ Technology Stack

## Programming Language

- Python


## Libraries

- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib


## Deployment

- Streamlit


## Development Environment

- Jupyter Notebook
- VS Code


---

# 📁 Project Structure

Customer-Churn-Prediction/

│
├── app.py
│
├── requirements.txt
│
├── README.md
│
├── models/
│ │
│ ├── logistic_regression_model.pkl
│ │
│ └── feature_columns.pkl
│
├── notebooks/
│ │
│ └── Customer_Churn_EDA.ipynb
│
└── dataset/
│
└── telecom_customer_data.csv



---

# ⚙️ Installation & Setup

## 1. Clone Repository

```bash
git clone <repository-link>

cd Customer-Churn-Prediction

pip install -r requirements.txt

streamlit==1.59.2
pandas==3.0.3
numpy==2.4.6
scikit-learn==1.9.0
joblib==1.5.3

📈 Business Impact

This system can help telecom companies:

Identify high-risk customers early
Reduce customer loss
Improve retention strategies
Increase customer lifetime value
Make data-driven decisions

🔮 Future Improvements

Possible improvements:

Deploy model on cloud platforms
Add customer dashboard analytics
Use advanced models like XGBoost and Random Forest
Add explainable AI using SHAP
Connect with real-time customer databases

👨‍💻 Author

Shreyash Sanjay Lohkare

B.Tech Data Science Student

Skills:

Python
Machine Learning
Data Analytics
Streamlit

⭐ If you like this project

Give this repository a star ⭐ and feel free to explore!


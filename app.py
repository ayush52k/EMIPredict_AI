import streamlit as st
import pandas as pd
import joblib


# ==========================================
# LOAD SAVED MODELS
# ==========================================

classification_model = joblib.load(
    "emi_classification_model.pkl"
)

regression_model = joblib.load(
    "emi_regression_model.pkl"
)


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="EMI Prediction System",
    page_icon="💰",
    layout="wide"
)


# ==========================================
# TITLE
# ==========================================

st.title("💰 EMI Prediction System")

st.write(
    "Enter the customer's financial information "
    "to predict EMI eligibility and maximum safe EMI."
)

st.divider()


# ==========================================
# CUSTOMER INFORMATION
# ==========================================

st.header("👤 Customer Information")

col1, col2, col3 = st.columns(3)

with col1:

    age = st.number_input(
        "Age",
        min_value=25,
        max_value=60,
        value=35
    )

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    marital_status = st.selectbox(
        "Marital Status",
        ["Single", "Married"]
    )

    education = st.selectbox(
        "Education",
        [
            "High School",
            "Graduate",
            "Post Graduate"
        ]
    )


with col2:

    employment_type = st.selectbox(
        "Employment Type",
        [
            "Salaried",
            "Self Employed",
            "Business"
        ]
    )

    company_type = st.selectbox(
        "Company Type",
        [
            "Private",
            "Government",
            "Public",
            "Other"
        ]
    )

    years_of_employment = st.number_input(
        "Years of Employment",
        min_value=0,
        max_value=40,
        value=5
    )

    house_type = st.selectbox(
        "House Type",
        [
            "Owned",
            "Rented",
            "Family"
        ]
    )


with col3:

    family_size = st.number_input(
        "Family Size",
        min_value=1,
        max_value=15,
        value=4
    )

    dependents = st.number_input(
        "Dependents",
        min_value=0,
        max_value=10,
        value=2
    )

    existing_loans = st.selectbox(
        "Existing Loans",
        [
            "Yes",
            "No"
        ]
    )

    emi_scenario = st.selectbox(
        "EMI Scenario",
        [
            "Normal",
            "High"
        ]
    )


# ==========================================
# INCOME INFORMATION
# ==========================================

st.divider()

st.header("💵 Income & Financial Information")

col1, col2, col3 = st.columns(3)

with col1:

    monthly_salary = st.number_input(
        "Monthly Salary (₹)",
        min_value=15000,
        max_value=200000,
        value=80000
    )

    monthly_rent = st.number_input(
        "Monthly Rent (₹)",
        min_value=0,
        max_value=100000,
        value=15000
    )

    school_fees = st.number_input(
        "School Fees (₹)",
        min_value=0,
        max_value=100000,
        value=5000
    )

    college_fees = st.number_input(
        "College Fees (₹)",
        min_value=0,
        max_value=100000,
        value=0
    )


with col2:

    travel_expenses = st.number_input(
        "Travel Expenses (₹)",
        min_value=0,
        max_value=50000,
        value=5000
    )

    groceries_utilities = st.number_input(
        "Groceries & Utilities (₹)",
        min_value=0,
        max_value=100000,
        value=10000
    )

    other_monthly_expenses = st.number_input(
        "Other Monthly Expenses (₹)",
        min_value=0,
        max_value=100000,
        value=5000
    )

    current_emi_amount = st.number_input(
        "Current EMI Amount (₹)",
        min_value=0,
        max_value=100000,
        value=5000
    )


with col3:

    credit_score = st.number_input(
        "Credit Score",
        min_value=300,
        max_value=900,
        value=750
    )

    bank_balance = st.number_input(
        "Bank Balance (₹)",
        min_value=0,
        max_value=1000000,
        value=100000
    )

    emergency_fund = st.number_input(
        "Emergency Fund (₹)",
        min_value=0,
        max_value=1000000,
        value=50000
    )


# ==========================================
# LOAN INFORMATION
# ==========================================

st.divider()

st.header("🏦 Loan Request")

col1, col2 = st.columns(2)

with col1:

    requested_amount = st.number_input(
        "Requested Loan Amount (₹)",
        min_value=10000,
        max_value=5000000,
        value=500000
    )


with col2:

    requested_tenure = st.number_input(
        "Requested Tenure (Months)",
        min_value=3,
        max_value=84,
        value=36
    )


# ==========================================
# FEATURE ENGINEERING
# ==========================================

total_monthly_expenses = (
    monthly_rent
    + school_fees
    + college_fees
    + travel_expenses
    + groceries_utilities
    + other_monthly_expenses
    + current_emi_amount
)

debt_to_income_ratio = (
    current_emi_amount / monthly_salary
)

expense_to_income_ratio = (
    total_monthly_expenses / monthly_salary
)

available_income = (
    monthly_salary - total_monthly_expenses
)


# We don't know max EMI yet because that's what
# the regression model will predict.

# Use a temporary value for classification.
# The original dataset contains max_monthly_emi
# as an input-derived feature.

affordability_ratio = 0


# ==========================================
# PREDICTION BUTTON
# ==========================================

st.divider()

predict_button = st.button(
    "🔮 PREDICT EMI",
    use_container_width=True
)


# ==========================================
# MAKE PREDICTION
# ==========================================

if predict_button:

    # --------------------------------------
    # CREATE INPUT DATAFRAME
    # --------------------------------------

    input_data = pd.DataFrame({
        "age": [age],
        "gender": [gender],
        "marital_status": [marital_status],
        "education": [education],
        "monthly_salary": [monthly_salary],
        "employment_type": [employment_type],
        "years_of_employment": [years_of_employment],
        "company_type": [company_type],
        "house_type": [house_type],
        "monthly_rent": [monthly_rent],
        "family_size": [family_size],
        "dependents": [dependents],
        "school_fees": [school_fees],
        "college_fees": [college_fees],
        "travel_expenses": [travel_expenses],
        "groceries_utilities": [groceries_utilities],
        "other_monthly_expenses": [other_monthly_expenses],
        "existing_loans": [existing_loans],
        "current_emi_amount": [current_emi_amount],
        "credit_score": [credit_score],
        "bank_balance": [bank_balance],
        "emergency_fund": [emergency_fund],
        "emi_scenario": [emi_scenario],
        "requested_amount": [requested_amount],
        "requested_tenure": [requested_tenure],
        "total_monthly_expenses": [total_monthly_expenses],
        "debt_to_income_ratio": [debt_to_income_ratio],
        "expense_to_income_ratio": [expense_to_income_ratio],
        "available_income": [available_income],
        "affordability_ratio": [affordability_ratio]
    })


    # --------------------------------------
    # REGRESSION PREDICTION FIRST
    # --------------------------------------

    predicted_emi = regression_model.predict(
        input_data
    )[0]

    # EMI cannot be negative
    predicted_emi = max(
        500,
        min(predicted_emi, 50000)
    )


    # --------------------------------------
    # UPDATE AFFORDABILITY RATIO
    # --------------------------------------

    affordability_ratio = (
        predicted_emi / monthly_salary
    )

    input_data["affordability_ratio"] = (
        affordability_ratio
    )


    # --------------------------------------
    # CLASSIFICATION PREDICTION
    # --------------------------------------

    eligibility = classification_model.predict(
        input_data
    )[0]


    # --------------------------------------
    # DISPLAY RESULTS
    # --------------------------------------

    st.divider()

    st.header("📊 Prediction Results")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Maximum Safe EMI",
            f"₹{predicted_emi:,.0f} / month"
        )


    with col2:

        if eligibility == "Eligible":

            st.success(
                "✅ ELIGIBLE"
            )

        elif eligibility == "High_Risk":

            st.warning(
                "⚠️ HIGH RISK"
            )

        else:

            st.error(
                "❌ NOT ELIGIBLE"
            )


    st.subheader("Financial Summary")

    summary = pd.DataFrame({
        "Metric": [
            "Monthly Salary",
            "Total Monthly Expenses",
            "Available Income",
            "Credit Score",
            "Requested Loan",
            "Requested Tenure"
        ],

        "Value": [
            f"₹{monthly_salary:,.0f}",
            f"₹{total_monthly_expenses:,.0f}",
            f"₹{available_income:,.0f}",
            credit_score,
            f"₹{requested_amount:,.0f}",
            f"{requested_tenure} months"
        ]
    })

    st.table(summary)
    
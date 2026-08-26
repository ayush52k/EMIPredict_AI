import pandas as pd
import numpy as np

# Make results reproducible
np.random.seed(42)

# Number of customers
n = 5000

# Create financial dataset
data = {
    "age": np.random.randint(25, 61, n),

    "gender": np.random.choice(
        ["Male", "Female"], n
    ),

    "marital_status": np.random.choice(
        ["Single", "Married"], n
    ),

    "education": np.random.choice(
        ["High School", "Graduate", "Post Graduate", "Professional"],
        n
    ),

    "monthly_salary": np.random.randint(
        15000, 200001, n
    ),

    "employment_type": np.random.choice(
        ["Private", "Government", "Self-employed"],
        n
    ),

    "years_of_employment": np.random.randint(
        0, 31, n
    ),

    "company_type": np.random.choice(
        ["Small", "Medium", "Large"],
        n
    ),

    "house_type": np.random.choice(
        ["Rented", "Own", "Family"],
        n
    ),

    "monthly_rent": np.random.randint(
        0, 30001, n
    ),

    "family_size": np.random.randint(
        1, 7, n
    ),

    "dependents": np.random.randint(
        0, 5, n
    ),

    "school_fees": np.random.randint(
        0, 15001, n
    ),

    "college_fees": np.random.randint(
        0, 30001, n
    ),

    "travel_expenses": np.random.randint(
        1000, 15001, n
    ),

    "groceries_utilities": np.random.randint(
        3000, 30001, n
    ),

    "other_monthly_expenses": np.random.randint(
        0, 20001, n
    ),

    "existing_loans": np.random.choice(
        ["Yes", "No"], n
    ),

    "current_emi_amount": np.random.randint(
        0, 30001, n
    ),

    "credit_score": np.random.randint(
        300, 851, n
    ),

    "bank_balance": np.random.randint(
        5000, 1000001, n
    ),

    "emergency_fund": np.random.randint(
        0, 500001, n
    ),

    "emi_scenario": np.random.choice(
        [
            "E-commerce Shopping",
            "Home Appliances",
            "Vehicle",
            "Personal Loan",
            "Education"
        ],
        n
    ),

    "requested_amount": np.random.randint(
        10000, 1500001, n
    ),

    "requested_tenure": np.random.randint(
        3, 85, n
    )
}

# Convert dictionary into a DataFrame
df = pd.DataFrame(data)

# Calculate total monthly expenses
total_expenses = (
    df["monthly_rent"]
    + df["school_fees"]
    + df["college_fees"]
    + df["travel_expenses"]
    + df["groceries_utilities"]
    + df["other_monthly_expenses"]
    + df["current_emi_amount"]
)

# Calculate approximate financial capacity
available_income = df["monthly_salary"] - total_expenses

# Maximum safe EMI
df["max_monthly_emi"] = np.clip(
    available_income * 0.35,
    500,
    50000
)

# Determine EMI eligibility
def calculate_eligibility(row):

    if (
        row["credit_score"] >= 700
        and row["max_monthly_emi"] >= 10000
    ):
        return "Eligible"

    elif (
        row["credit_score"] >= 600
        and row["max_monthly_emi"] >= 5000
    ):
        return "High_Risk"

    else:
        return "Not_Eligible"


df["emi_eligibility"] = df.apply(
    calculate_eligibility,
    axis=1
)

# Save dataset
df.to_csv(
    "EMI_dataset.csv",
    index=False
)

print("Dataset created successfully!")
print("Number of records:", len(df))
print("Number of columns:", len(df.columns))
print("\nFirst 5 records:")
print(df.head())

print("\nEligibility distribution:")
print(df["emi_eligibility"].value_counts())
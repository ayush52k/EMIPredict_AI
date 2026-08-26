import pandas as pd

# Load dataset
df = pd.read_csv("EMI_dataset.csv")

print("Original columns:", len(df.columns))

# -----------------------------------
# FEATURE ENGINEERING
# -----------------------------------

# Total monthly expenses
df["total_monthly_expenses"] = (
    df["monthly_rent"]
    + df["school_fees"]
    + df["college_fees"]
    + df["travel_expenses"]
    + df["groceries_utilities"]
    + df["other_monthly_expenses"]
    + df["current_emi_amount"]
)

# Debt-to-income ratio
df["debt_to_income_ratio"] = (
    df["current_emi_amount"] /
    df["monthly_salary"]
)

# Expense-to-income ratio
df["expense_to_income_ratio"] = (
    df["total_monthly_expenses"] /
    df["monthly_salary"]
)

# Available income
df["available_income"] = (
    df["monthly_salary"] -
    df["total_monthly_expenses"]
)

# Affordability ratio
df["affordability_ratio"] = (
    df["max_monthly_emi"] /
    df["monthly_salary"]
)

# -----------------------------------
# DISPLAY RESULTS
# -----------------------------------

print("\n========== NEW FEATURES ==========")

print(df[
    [
        "monthly_salary",
        "current_emi_amount",
        "total_monthly_expenses",
        "debt_to_income_ratio",
        "expense_to_income_ratio",
        "available_income",
        "affordability_ratio"
    ]
].head())

print("\n========== DATASET SIZE ==========")

print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

# Save processed dataset
df.to_csv(
    "EMI_processed.csv",
    index=False
)

print("\nProcessed dataset saved successfully!")
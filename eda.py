import pandas as pd

# Load our dataset
df = pd.read_csv("EMI_dataset.csv")

print("========== DATASET INFORMATION ==========")

# Number of rows and columns
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\n========== COLUMN NAMES ==========")
print(df.columns.tolist())

print("\n========== FIRST 5 ROWS ==========")
print(df.head())

print("\n========== DATA TYPES ==========")
print(df.dtypes)

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

print("\n========== DUPLICATE ROWS ==========")
print(df.duplicated().sum())

print("\n========== ELIGIBILITY DISTRIBUTION ==========")
print(df["emi_eligibility"].value_counts())

print("\n========== BASIC STATISTICS ==========")
print(df.describe())

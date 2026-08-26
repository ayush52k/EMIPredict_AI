import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv("EMI_processed.csv")

print("Dataset loaded!")
print("Rows:", len(df))


# ==========================================
# 2. SEPARATE INPUTS AND TARGET
# ==========================================

# X = information about the customer
X = df.drop(
    columns=["emi_eligibility", "max_monthly_emi"]
)

# y = answer we want the model to predict
y = df["emi_eligibility"]


# ==========================================
# 3. FIND CATEGORICAL AND NUMERICAL COLUMNS
# ==========================================

categorical_columns = X.select_dtypes(
    include=["object"]
).columns

numerical_columns = X.select_dtypes(
    exclude=["object"]
).columns


print("\nCategorical columns:")
print(list(categorical_columns))

print("\nNumerical columns:")
print(list(numerical_columns))


# ==========================================
# 4. CONVERT TEXT DATA INTO NUMBERS
# ==========================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_columns
        )
    ],
    remainder="passthrough"
)


# ==========================================
# 5. CREATE RANDOM FOREST MODEL
# ==========================================

model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),

        (
            "classifier",
            RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                class_weight="balanced"
            )
        )
    ]
)


# ==========================================
# 6. SPLIT DATA
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# ==========================================
# 7. TRAIN RANDOM FOREST
# ==========================================

print("\nTraining Random Forest...")

model.fit(
    X_train,
    y_train
)

print("Training completed!")


# ==========================================
# 8. MAKE PREDICTIONS
# ==========================================

y_pred = model.predict(X_test)


# ==========================================
# 9. EVALUATE MODEL
# ==========================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n========== RANDOM FOREST RESULTS ==========")

print(
    "Accuracy:",
    round(accuracy * 100, 2),
    "%"
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred
    )
)
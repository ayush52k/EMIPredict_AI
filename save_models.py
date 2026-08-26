import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor


# ==========================================
# LOAD DATA
# ==========================================

df = pd.read_csv("EMI_processed.csv")

X = df.drop(
    columns=["emi_eligibility", "max_monthly_emi"]
)

y_classification = df["emi_eligibility"]

y_regression = df["max_monthly_emi"]


# ==========================================
# CATEGORICAL COLUMNS
# ==========================================

categorical_columns = X.select_dtypes(
    include=["object"]
).columns


# ==========================================
# PREPROCESSOR
# ==========================================

preprocessor_classification = ColumnTransformer(
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
# CLASSIFICATION MODEL
# ==========================================

classification_model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor_classification
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


print("Training classification model...")

classification_model.fit(
    X,
    y_classification
)

print("Classification model trained!")


# ==========================================
# REGRESSION PREPROCESSOR
# ==========================================

preprocessor_regression = ColumnTransformer(
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
# REGRESSION MODEL
# ==========================================

regression_model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor_regression
        ),
        (
            "regressor",
            RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)


print("Training regression model...")

regression_model.fit(
    X,
    y_regression
)

print("Regression model trained!")


# ==========================================
# SAVE MODELS
# ==========================================

joblib.dump(
    classification_model,
    "emi_classification_model.pkl"
)

joblib.dump(
    regression_model,
    "emi_regression_model.pkl"
)


print("\n================================")
print("MODELS SAVED SUCCESSFULLY!")
print("================================")

print("emi_classification_model.pkl")
print("emi_regression_model.pkl")
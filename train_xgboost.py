import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

from xgboost import XGBClassifier


# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv("EMI_processed.csv")

print("Dataset loaded!")
print("Rows:", len(df))


# ==========================================
# 2. SEPARATE INPUTS AND TARGET
# ==========================================

X = df.drop(
    columns=["emi_eligibility", "max_monthly_emi"]
)

y = df["emi_eligibility"]


# ==========================================
# 3. CONVERT TARGET CLASSES TO NUMBERS
# ==========================================

# XGBoost needs numerical target values

target_mapping = {
    "Not_Eligible": 0,
    "High_Risk": 1,
    "Eligible": 2
}

y = y.map(target_mapping)


# ==========================================
# 4. FIND CATEGORICAL COLUMNS
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
# 5. CONVERT CATEGORICAL DATA
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
# 6. CREATE XGBOOST MODEL
# ==========================================

model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),

        (
            "classifier",
            XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                eval_metric="mlogloss"
            )
        )
    ]
)


# ==========================================
# 7. SPLIT DATA
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
# 8. TRAIN XGBOOST
# ==========================================

print("\nTraining XGBoost...")

model.fit(
    X_train,
    y_train
)

print("Training completed!")


# ==========================================
# 9. MAKE PREDICTIONS
# ==========================================

y_pred = model.predict(X_test)


# ==========================================
# 10. EVALUATE MODEL
# ==========================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n========== XGBOOST RESULTS ==========")

print(
    "Accuracy:",
    round(accuracy * 100, 2),
    "%"
)


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Not_Eligible",
            "High_Risk",
            "Eligible"
        ]
    )
)
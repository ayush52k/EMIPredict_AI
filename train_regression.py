import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

import numpy as np


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
    columns=["max_monthly_emi", "emi_eligibility"]
)

y = df["max_monthly_emi"]


# ==========================================
# 3. FIND CATEGORICAL COLUMNS
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
# 5. CREATE RANDOM FOREST REGRESSOR
# ==========================================

model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
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


# ==========================================
# 6. SPLIT DATA
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# ==========================================
# 7. TRAIN MODEL
# ==========================================

print("\nTraining Random Forest Regressor...")

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
# 9. CALCULATE METRICS
# ==========================================

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        y_pred
    )
)

mae = mean_absolute_error(
    y_test,
    y_pred
)

r2 = r2_score(
    y_test,
    y_pred
)


# ==========================================
# 10. DISPLAY RESULTS
# ==========================================

print("\n========== RANDOM FOREST REGRESSION RESULTS ==========")

print(
    "RMSE:",
    round(rmse, 2)
)

print(
    "MAE:",
    round(mae, 2)
)

print(
    "R² Score:",
    round(r2, 4)
)


# ==========================================
# 11. SAMPLE PREDICTIONS
# ==========================================

results = pd.DataFrame({
    "Actual EMI": y_test.values[:10],
    "Predicted EMI": y_pred[:10]
})

print("\n========== SAMPLE PREDICTIONS ==========")

print(results)
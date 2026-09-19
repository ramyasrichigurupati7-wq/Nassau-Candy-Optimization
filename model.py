import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# ==========================================
# 1. LOAD CLEANED DATASET
# ==========================================

print("======================================")
print("LOADING CLEANED DATASET")
print("======================================")

df = pd.read_csv("cleaned_nassau_candy.csv")

print("Dataset loaded successfully!")
print("Rows:", len(df))
print("Columns:", len(df.columns))


# ==========================================
# 2. REMOVE MISSING VALUES
# ==========================================

df = df.dropna(
    subset=[
        "Shipping Days",
        "Product Name",
        "Region",
        "Ship Mode",
        "Units",
        "Sales",
        "Cost"
    ]
)


# ==========================================
# 3. SELECT FEATURES
# ==========================================

features = [
    "Product Name",
    "Region",
    "Ship Mode",
    "Units",
    "Sales",
    "Cost"
]

target = "Shipping Days"


X = df[features]

y = df[target]


# ==========================================
# 4. CATEGORICAL AND NUMERICAL FEATURES
# ==========================================

categorical_features = [
    "Product Name",
    "Region",
    "Ship Mode"
]

numerical_features = [
    "Units",
    "Sales",
    "Cost"
]


# ==========================================
# 5. PREPROCESSING
# ==========================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        ),
        (
            "numerical",
            "passthrough",
            numerical_features
        )
    ]
)


# ==========================================
# 6. TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# ==========================================
# 7. LINEAR REGRESSION MODEL
# ==========================================

linear_model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            LinearRegression()
        )
    ]
)


print("\nTraining Linear Regression...")

linear_model.fit(
    X_train,
    y_train
)


linear_predictions = linear_model.predict(
    X_test
)


# ==========================================
# 8. RANDOM FOREST MODEL
# ==========================================

random_forest_model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            RandomForestRegressor(
                n_estimators=100,
                random_state=42
            )
        )
    ]
)


print("Training Random Forest...")

random_forest_model.fit(
    X_train,
    y_train
)


rf_predictions = random_forest_model.predict(
    X_test
)


# ==========================================
# 9. MODEL EVALUATION
# ==========================================

print("\n======================================")
print("MODEL RESULTS")
print("======================================")


# Linear Regression

linear_mae = mean_absolute_error(
    y_test,
    linear_predictions
)

linear_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        linear_predictions
    )
)

linear_r2 = r2_score(
    y_test,
    linear_predictions
)


print("\nLinear Regression")

print("MAE:", round(linear_mae, 3))

print("RMSE:", round(linear_rmse, 3))

print("R2 Score:", round(linear_r2, 3))


# Random Forest

rf_mae = mean_absolute_error(
    y_test,
    rf_predictions
)

rf_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        rf_predictions
    )
)

rf_r2 = r2_score(
    y_test,
    rf_predictions
)


print("\nRandom Forest")

print("MAE:", round(rf_mae, 3))

print("RMSE:", round(rf_rmse, 3))

print("R2 Score:", round(rf_r2, 3))


# ==========================================
# 10. SAVE RANDOM FOREST MODEL
# ==========================================

model_path = "shipping_model.pkl"

joblib.dump(
    random_forest_model,
    model_path
)


print("\n======================================")
print("MODEL SAVED SUCCESSFULLY")
print("======================================")

print("File created:", model_path)


# ==========================================
# 11. FINAL MESSAGE
# ==========================================

print("\nMachine Learning Model Completed!")
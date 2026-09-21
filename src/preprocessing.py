"""Reusable preprocessing for customer churn models."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    StandardScaler,
)

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]

CATEGORICAL_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def clean_numeric(values):
    """Convert numeric strings; represent missing or invalid values as NaN."""
    frame = pd.DataFrame(values)
    converted = frame.apply(pd.to_numeric, errors="coerce")
    return converted.to_numpy(dtype=float, na_value=np.nan)


def clean_categories(values):
    """Strip whitespace and normalize missing categories."""
    frame = pd.DataFrame(values)
    cleaned = frame.apply(lambda column: column.astype("string").str.strip())
    cleaned = cleaned.replace("", pd.NA)
    return cleaned.to_numpy(dtype=object, na_value=np.nan)


def build_preprocessor() -> ColumnTransformer:
    """Create an unfitted transformer; learning happens only during fit."""
    numeric_pipeline = Pipeline(
        steps=[
            (
                "clean",
                FunctionTransformer(
                    clean_numeric,
                    feature_names_out="one-to-one",
                ),
            ),
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "clean",
                FunctionTransformer(
                    clean_categories,
                    feature_names_out="one-to-one",
                ),
            ),
            ("impute", SimpleImputer(strategy="most_frequent")),
            (
                "encode",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
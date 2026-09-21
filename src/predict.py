"""Validate customer inputs and predict using the saved pipeline."""

import math
from pathlib import Path

import joblib
import pandas as pd

from src.data import PROJECT_ROOT
from src.preprocessing import FEATURES

MODEL_PATH = PROJECT_ROOT / "models" / "churn_pipeline.joblib"

INTERNET_ADDONS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

CATEGORY_OPTIONS = {
    "gender": ["Female", "Male"],
    "Partner": ["No", "Yes"],
    "Dependents": ["No", "Yes"],
    "PhoneService": ["No", "Yes"],
    "MultipleLines": ["No", "Yes", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["No", "Yes"],
    "PaymentMethod": [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
    **{
        field: ["No", "Yes", "No internet service"]
        for field in INTERNET_ADDONS
    },
}


def load_artifact(path: Path = MODEL_PATH) -> dict:
    """Load our trusted local model artifact; never load user-uploaded models."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Saved model not found: {path}")

    artifact = joblib.load(path)
    if not {"pipeline", "threshold", "metadata"}.issubset(artifact):
        raise ValueError("Saved artifact is missing required fields.")

    threshold = float(artifact["threshold"])
    if not math.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError("Saved threshold must be between zero and one.")

    if artifact["metadata"]["features"] != FEATURES:
        raise ValueError("Saved model feature schema does not match the application.")

    return artifact


def validate_customer(customer: dict) -> pd.DataFrame:
    """Validate one customer and return model inputs in a consistent order."""
    missing = set(FEATURES) - set(customer)
    extra = set(customer) - set(FEATURES)

    if missing:
        raise ValueError(f"Missing customer fields: {', '.join(sorted(missing))}")
    if extra:
        raise ValueError(f"Unexpected customer fields: {', '.join(sorted(extra))}")

    values = dict(customer)

    for field in ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]:
        raw = values[field]

        # An explicitly unknown total is handled by the trained imputer.
        if field == "TotalCharges" and (
            raw is None or (isinstance(raw, str) and not raw.strip())
        ):
            values[field] = float("nan")
            continue

        if isinstance(raw, bool):
            raise ValueError(f"{field} must be a number, not a boolean.")

        try:
            number = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field} must be a valid number.") from exc

        if not math.isfinite(number) or number < 0:
            raise ValueError(f"{field} must be a finite, non-negative number.")

        if field in ["tenure", "SeniorCitizen"]:
            if not number.is_integer():
                raise ValueError(f"{field} must be a whole number.")
            values[field] = int(number)
        else:
            values[field] = number

    if values["SeniorCitizen"] not in [0, 1]:
        raise ValueError("SeniorCitizen must be 0 or 1.")

    for field, options in CATEGORY_OPTIONS.items():
        value = values[field]
        if not isinstance(value, str) or value.strip() not in options:
            raise ValueError(f"{field} must be one of: {', '.join(options)}")
        values[field] = value.strip()

    if values["PhoneService"] == "No":
        if values["MultipleLines"] != "No phone service":
            raise ValueError("Without phone service, use 'No phone service' for MultipleLines.")
    elif values["MultipleLines"] == "No phone service":
        raise ValueError("With phone service, MultipleLines must be Yes or No.")

    for field in INTERNET_ADDONS:
        if values["InternetService"] == "No":
            if values[field] != "No internet service":
                raise ValueError(f"Without internet service, {field} must be 'No internet service'.")
        elif values[field] == "No internet service":
            raise ValueError(f"With internet service, {field} must be Yes or No.")

    return pd.DataFrame([values], columns=FEATURES)


def predict_customer(customer: dict, artifact: dict) -> dict:
    """Return churn probability and a decision using the saved threshold."""
    inputs = validate_customer(customer)
    pipeline = artifact["pipeline"]

    classes = list(pipeline.named_steps["classifier"].classes_)
    if 1 not in classes:
        raise ValueError("Saved classifier has no positive churn class.")

    probability = float(pipeline.predict_proba(inputs)[0, classes.index(1)])
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("Model returned an invalid probability.")

    threshold = float(artifact["threshold"])
    churn = probability >= threshold

    return {
        "prediction": "Churn" if churn else "No churn",
        "churn_probability": probability,
        "threshold": threshold,
        "model": artifact["metadata"]["model"],
    }
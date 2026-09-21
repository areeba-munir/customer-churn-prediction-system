"""Test customer validation and saved-model prediction."""

import numpy as np
import pytest

from src.predict import load_artifact, predict_customer, validate_customer
from src.preprocessing import FEATURES


@pytest.fixture
def customer():
    """A realistic example used for input validation, not training."""
    return {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "Yes",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 55.0,
        "TotalCharges": 660.0,
    }


def test_valid_customer_schema_and_unknown_total(customer):
    customer["TotalCharges"] = None
    result = validate_customer(customer)

    assert result.shape == (1, 19)
    assert result.columns.tolist() == FEATURES
    assert np.isnan(result.loc[0, "TotalCharges"])
    assert customer["TotalCharges"] is None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("tenure", -1),
        ("tenure", 1.5),
        ("MonthlyCharges", float("inf")),
        ("TotalCharges", "invalid"),
        ("SeniorCitizen", 2),
        ("Contract", "Unknown contract"),
        ("PhoneService", "No"),
        ("InternetService", "No"),
    ],
)
def test_invalid_customer_is_rejected(customer, field, value):
    customer[field] = value

    with pytest.raises(ValueError):
        validate_customer(customer)


def test_missing_field_is_rejected(customer):
    del customer["Contract"]

    with pytest.raises(ValueError, match="Missing customer fields"):
        validate_customer(customer)


def test_missing_model_has_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="Saved model not found"):
        load_artifact(tmp_path / "missing.joblib")


def test_saved_model_prediction_uses_saved_threshold(customer):
    artifact = load_artifact()
    result = predict_customer(customer, artifact)

    inputs = validate_customer(customer)
    pipeline = artifact["pipeline"]
    positive_index = list(pipeline.named_steps["classifier"].classes_).index(1)
    expected_probability = pipeline.predict_proba(inputs)[0, positive_index]

    assert result["churn_probability"] == pytest.approx(expected_probability)
    assert 0 <= result["churn_probability"] <= 1
    assert result["threshold"] == artifact["threshold"]

    expected_label = (
        "Churn" if expected_probability >= artifact["threshold"] else "No churn"
    )
    assert result["prediction"] == expected_label
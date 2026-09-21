"""Tests for customer separation and reproducible splitting."""

import pandas as pd
import pytest

from src.split import split_customers


def example_customers():
    """Provide enough examples of both classes for a stratified split."""
    return pd.DataFrame(
        {
            "customerID": [f"customer-{i}" for i in range(50)],
            "Churn": ["No"] * 30 + ["Yes"] * 20,
        }
    )


def test_split_is_separate_complete_and_reproducible():
    data = example_customers()
    original = data.copy(deep=True)

    train, test = split_customers(data)
    repeat_train, repeat_test = split_customers(data)

    assert len(train) == 40
    assert len(test) == 10
    assert set(train["customerID"]).isdisjoint(test["customerID"])
    assert set(train["customerID"]) | set(test["customerID"]) == set(data["customerID"])
    assert train["Churn"].value_counts().to_dict() == {"No": 24, "Yes": 16}
    assert test["Churn"].value_counts().to_dict() == {"No": 6, "Yes": 4}

    pd.testing.assert_frame_equal(train, repeat_train)
    pd.testing.assert_frame_equal(test, repeat_test)
    pd.testing.assert_frame_equal(data, original)


def test_split_rejects_duplicate_customers():
    data = example_customers()
    data.loc[1, "customerID"] = data.loc[0, "customerID"]

    with pytest.raises(ValueError, match="must be unique"):
        split_customers(data)


def test_split_rejects_unknown_target():
    data = example_customers()
    data.loc[0, "Churn"] = "Unknown"

    with pytest.raises(ValueError, match="Churn must contain"):
        split_customers(data)
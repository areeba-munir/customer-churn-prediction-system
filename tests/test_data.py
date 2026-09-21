"""Tests for loading raw customer data."""

import pandas as pd
import pytest

from src.data import load_raw_data


def test_load_preserves_records_and_raw_file(tmp_path):
    """Loading must preserve identifiers, blank values, and source bytes."""
    dataset_path = tmp_path / "customers.csv"
    dataset_path.write_text(
        "customerID,TotalCharges,Churn\n"
        "001-A,29.85,No\n"
        "002-B, ,Yes\n",
        encoding="utf-8",
    )
    original_bytes = dataset_path.read_bytes()

    result = load_raw_data(dataset_path)

    assert isinstance(result, pd.DataFrame)
    assert result.shape == (2, 3)
    assert result.columns.tolist() == ["customerID", "TotalCharges", "Churn"]
    assert result["customerID"].tolist() == ["001-A", "002-B"]
    assert result["Churn"].tolist() == ["No", "Yes"]
    assert result.loc[1, "TotalCharges"] == " "
    assert dataset_path.read_bytes() == original_bytes


def test_load_missing_file_raises_clear_error(tmp_path):
    """A missing dataset must produce an understandable error."""
    missing_path = tmp_path / "missing.csv"

    with pytest.raises(FileNotFoundError, match="Dataset not found"):
        load_raw_data(missing_path)
"""Check preprocessing behavior on small controlled examples."""

import numpy as np
import pandas as pd

from src.preprocessing import CATEGORICAL_FEATURES, build_preprocessor


def example_customers():
    data = pd.DataFrame(
        {
            "tenure": [1, 12, 24],
            "MonthlyCharges": [20.0, 50.0, 80.0],
            "TotalCharges": ["20", " ", "1920"],
        }
    )
    for column in CATEGORICAL_FEATURES:
        data[column] = ["A", "B", "A"]
    data["SeniorCitizen"] = [0, 1, 0]
    return data


def test_numeric_imputation_and_feature_names():
    data = example_customers()
    preprocessor = build_preprocessor()

    result = preprocessor.fit_transform(data)

    assert result.shape[0] == 3
    assert np.isfinite(result).all()
    assert result.shape[1] == len(preprocessor.get_feature_names_out())

    imputer = preprocessor.named_transformers_["numeric"].named_steps["impute"]
    np.testing.assert_allclose(imputer.statistics_, [12, 50, 970])


def test_unknown_categories_and_missing_values_use_training_statistics():
    preprocessor = build_preprocessor()
    preprocessor.fit(example_customers())

    imputer = preprocessor.named_transformers_["numeric"].named_steps["impute"]
    original_statistics = imputer.statistics_.copy()

    new_customer = example_customers().iloc[[0]].copy()
    new_customer.loc[:, "Contract"] = "Previously unseen contract"
    new_customer.loc[:, "PaymentMethod"] = " "
    new_customer.loc[:, "TotalCharges"] = "not a number"

    result = preprocessor.transform(new_customer)

    assert np.isfinite(result).all()
    np.testing.assert_array_equal(imputer.statistics_, original_statistics)


def test_ids_targets_and_input_column_order_do_not_change_features():
    data = example_customers()
    preprocessor = build_preprocessor()
    expected = preprocessor.fit_transform(data)

    extra_columns = data.assign(customerID=["x", "y", "z"], Churn=["Yes", "No", "Yes"])
    reordered = extra_columns[extra_columns.columns[::-1]]

    actual = preprocessor.transform(reordered)

    np.testing.assert_allclose(actual, expected)
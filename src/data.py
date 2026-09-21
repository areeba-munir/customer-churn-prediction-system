"""Inspect the raw customer dataset without modifying it."""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"


def load_raw_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load customer records from a CSV file."""
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")

    return pd.read_csv(path)


def main() -> None:
    """Display initial data-quality checks."""
    df = load_raw_data()

    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    print("\nMissing values recognized by Pandas:")
    print(df.isna().sum().to_string())

    # A space-only cell may not be recognized as missing when loading.
    blank_counts = df.apply(
        lambda column: column.astype("string").str.strip().eq("").sum()
    )
    print("\nEmpty or whitespace-only values:")
    print(blank_counts.to_string())

    print(f"\nExact duplicate rows: {df.duplicated().sum()}")
    print(f"Missing customer IDs: {df['customerID'].isna().sum()}")
    print(f"Duplicate customer IDs: {df['customerID'].duplicated().sum()}")

    # Inspect conversion without changing the original column.
    total_text = df["TotalCharges"].astype("string").str.strip()
    total_numeric = pd.to_numeric(total_text, errors="coerce")

    blank_or_missing = total_text.isna() | total_text.eq("")
    invalid_nonblank = total_numeric.isna() & ~blank_or_missing

    print(f"\nTotalCharges blank or missing: {blank_or_missing.sum()}")
    print(f"TotalCharges invalid nonblank values: {invalid_nonblank.sum()}")

    numeric_fields = pd.DataFrame(
        {
            "tenure": df["tenure"],
            "MonthlyCharges": df["MonthlyCharges"],
            "TotalCharges": total_numeric,
        }
    )

    print("\nNumerical summary:")
    print(numeric_fields.describe().to_string())

    print("\nNegative-value counts:")
    print(numeric_fields.lt(0).sum().to_string())

    print("\nSeniorCitizen values:")
    print(df["SeniorCitizen"].value_counts(dropna=False).to_string())

    print("\nChurn counts:")
    print(df["Churn"].value_counts(dropna=False).to_string())

    print("\nChurn percentages:")
    print(
        df["Churn"]
        .value_counts(normalize=True, dropna=False)
        .mul(100)
        .round(2)
        .to_string()
    )


if __name__ == "__main__":
    main()
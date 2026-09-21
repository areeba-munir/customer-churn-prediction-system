"""Create reproducible training and final-test datasets."""

import hashlib

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data import DATA_PATH, PROJECT_ROOT, load_raw_data

RANDOM_STATE = 42
TEST_SIZE = 0.20


def split_customers(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split unique customers while preserving churn proportions."""
    required = {"customerID", "Churn"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    customer_ids = df["customerID"].astype("string")
    if customer_ids.isna().any() or customer_ids.str.strip().eq("").any():
        raise ValueError("Customer IDs must not be missing or blank.")

    if customer_ids.duplicated().any():
        raise ValueError("Customer IDs must be unique before splitting.")

    if df["Churn"].isna().any() or set(df["Churn"].unique()) != {"Yes", "No"}:
        raise ValueError("Churn must contain both Yes and No, with no missing values.")

    train, test = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["Churn"],
    )
    return train.copy(), test.copy()


def main() -> None:
    """Save the split and a short reproducibility record."""
    df = load_raw_data()
    train, test = split_customers(df)

    output_dir = PROJECT_ROOT / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    train.to_csv(output_dir / "train.csv", index=False)
    test.to_csv(output_dir / "test.csv", index=False)

    overlap = set(train["customerID"]) & set(test["customerID"])
    raw_hash = hashlib.sha256(DATA_PATH.read_bytes()).hexdigest()

    summary = (
        f"Random seed: {RANDOM_STATE}\n"
        f"Requested test fraction: {TEST_SIZE}\n"
        f"Raw dataset SHA-256: {raw_hash}\n"
        f"Total customers: {len(df)}\n"
        f"Training customers: {len(train)}\n"
        f"Final-test customers: {len(test)}\n"
        f"Overlapping customer IDs: {len(overlap)}\n"
        "Use training data only for EDA, tuning, and model selection.\n"
    )

    report_dir = PROJECT_ROOT / "reports" / "metrics"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "split_summary.txt").write_text(summary, encoding="utf-8")

    print(summary)


if __name__ == "__main__":
    main()
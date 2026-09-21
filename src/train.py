"""Evaluate a baseline churn model using training-data cross-validation."""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from src.data import PROJECT_ROOT
from src.preprocessing import FEATURES, build_preprocessor

RANDOM_STATE = 42


def build_baseline() -> Pipeline:
    """Combine preprocessing and Logistic Regression in one pipeline."""
    return Pipeline(
        steps=[
            ("preprocessing", build_preprocessor()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def main() -> None:
    """Evaluate the baseline and save actual cross-validation results."""
    train_path = PROJECT_ROOT / "data" / "processed" / "train.csv"
    data = pd.read_csv(train_path)

    X = data.loc[:, FEATURES]
    y = data["Churn"].map({"No": 0, "Yes": 1})

    if y.isna().any() or y.nunique() != 2:
        raise ValueError("Training data must contain valid Yes and No churn labels.")

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }

    print(f"Training customers: {len(X)}")
    print("Running five-fold cross-validation...")
    print("The final test set is not loaded.")

    scores = cross_validate(
        build_baseline(),
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=1,
        error_score="raise",
    )

    fold_results = pd.DataFrame(
        {metric: scores[f"test_{metric}"] for metric in scoring}
    )
    fold_results.index = pd.RangeIndex(1, len(fold_results) + 1, name="fold")

    summary = fold_results.agg(["mean", "std"]).T
    summary.index.name = "metric"

    output_dir = PROJECT_ROOT / "reports" / "metrics"
    output_dir.mkdir(parents=True, exist_ok=True)

    fold_results.to_csv(output_dir / "baseline_cv_folds.csv")
    summary.to_csv(output_dir / "baseline_cv_summary.csv")

    print("\nCross-validation results by fold:")
    print(fold_results.round(4).to_string())

    print("\nMean and standard deviation across folds:")
    print(summary.round(4).to_string())

    print("\nPositive class: churn (Yes = 1).")
    print("Classification metrics use the classifier's default decision rule.")
    print("These are validation results, not final holdout-test results.")


if __name__ == "__main__":
    main()
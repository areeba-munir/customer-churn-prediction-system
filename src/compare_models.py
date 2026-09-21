"""Compare churn classifiers using identical training-data folds."""

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from src.data import PROJECT_ROOT
from src.preprocessing import FEATURES, build_preprocessor

RANDOM_STATE = 42

SCORING = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}


def build_candidates() -> dict:
    """Create unfitted candidate classifiers."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            random_state=RANDOM_STATE,
        ),
        "Balanced Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=RANDOM_STATE,
        ),
    }


def main() -> None:
    """Evaluate candidates without accessing the final test set."""
    data = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "train.csv")
    X = data.loc[:, FEATURES]
    y = data["Churn"].map({"No": 0, "Yes": 1})

    if y.isna().any() or y.nunique() != 2:
        raise ValueError("Training data must contain valid Yes and No churn labels.")

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    # Reuse exactly the same row assignments for every candidate.
    folds = list(cv.split(X, y))

    summary_rows = []
    all_fold_results = []

    for name, classifier in build_candidates().items():
        print(f"\nEvaluating: {name}", flush=True)

        pipeline = Pipeline(
            steps=[
                ("preprocessing", build_preprocessor()),
                ("classifier", classifier),
            ]
        )

        scores = cross_validate(
            pipeline,
            X,
            y,
            cv=folds,
            scoring=SCORING,
            n_jobs=1,
            error_score="raise",
        )

        fold_results = pd.DataFrame(
            {metric: scores[f"test_{metric}"] for metric in SCORING}
        )
        fold_results.insert(0, "fold", range(1, len(fold_results) + 1))
        fold_results.insert(0, "model", name)
        all_fold_results.append(fold_results)

        row = {"model": name}
        for metric in SCORING:
            row[f"{metric}_mean"] = fold_results[metric].mean()
            row[f"{metric}_std"] = fold_results[metric].std()
        summary_rows.append(row)

        print(
            f"Recall: {row['recall_mean']:.4f} | "
            f"Precision: {row['precision_mean']:.4f} | "
            f"ROC-AUC: {row['roc_auc_mean']:.4f}",
            flush=True,
        )

    comparison = pd.DataFrame(summary_rows)
    output_dir = PROJECT_ROOT / "reports" / "metrics"
    output_dir.mkdir(parents=True, exist_ok=True)

    comparison.to_csv(output_dir / "model_comparison.csv", index=False)
    pd.concat(all_fold_results, ignore_index=True).to_csv(
        output_dir / "model_comparison_folds.csv",
        index=False,
    )

    display_columns = ["model"] + [f"{metric}_mean" for metric in SCORING]
    print("\nMean five-fold validation results:")
    print(comparison[display_columns].round(4).to_string(index=False))

    print("\nStandard deviations are included in model_comparison.csv.")
    print("Positive class: churn (Yes = 1).")
    print("All candidates use their default classification decision rule.")
    print("No final model or threshold has been selected.")
    print("The final test set was not loaded.")


if __name__ == "__main__":
    main()
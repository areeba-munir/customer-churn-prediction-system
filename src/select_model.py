"""Select a model and threshold using training-only validation predictions."""

import json

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from src.compare_models import RANDOM_STATE, build_candidates
from src.data import PROJECT_ROOT
from src.preprocessing import FEATURES, build_preprocessor

MIN_PRECISION = 0.50
THRESHOLDS = np.round(np.arange(0.10, 0.91, 0.01), 2)


def main() -> None:
    """Compare thresholds without loading the final test set."""
    output_dir = PROJECT_ROOT / "reports" / "metrics"
    output_dir.mkdir(parents=True, exist_ok=True)
    selection_path = output_dir / "selected_model.json"

    if selection_path.exists():
        raise FileExistsError(
            "A model selection already exists. Review it before running selection again."
        )

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
    folds = list(cv.split(X, y))
    rows = []

    for name, classifier in build_candidates().items():
        print(f"Generating validation probabilities: {name}", flush=True)

        pipeline = Pipeline(
            steps=[
                ("preprocessing", build_preprocessor()),
                ("classifier", classifier),
            ]
        )

        # Every customer is predicted by a model trained without that customer.
        probabilities = cross_val_predict(
            pipeline,
            X,
            y,
            cv=folds,
            method="predict_proba",
            n_jobs=1,
        )[:, 1]

        auc = roc_auc_score(y, probabilities)

        for threshold in THRESHOLDS:
            predictions = (probabilities >= threshold).astype(int)

            rows.append(
                {
                    "model": name,
                    "threshold": float(threshold),
                    "accuracy": accuracy_score(y, predictions),
                    "precision": precision_score(y, predictions, zero_division=0),
                    "recall": recall_score(y, predictions, zero_division=0),
                    "f1": f1_score(y, predictions, zero_division=0),
                    "roc_auc": auc,
                }
            )

    results = pd.DataFrame(rows)
    results.to_csv(output_dir / "threshold_comparison.csv", index=False)

    eligible = results.loc[results["precision"] >= MIN_PRECISION].copy()
    if eligible.empty:
        raise ValueError("No candidate meets the minimum precision criterion.")

    # Break equal-recall ties using precision, F1, and then model name/threshold.
    ranked = eligible.sort_values(
        ["recall", "precision", "f1", "model", "threshold"],
        ascending=[False, False, False, True, True],
    )

    best_per_model = ranked.drop_duplicates("model")
    best_per_model.to_csv(output_dir / "best_thresholds.csv", index=False)

    winner = ranked.iloc[0]
    selection = {
        "model": str(winner["model"]),
        "threshold": float(winner["threshold"]),
        "random_state": RANDOM_STATE,
        "cv_folds": 5,
        "minimum_precision": MIN_PRECISION,
        "selection_rule": "Maximize pooled OOF recall subject to precision >= 0.50",
        "validation_metrics": {
            metric: float(winner[metric])
            for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]
        },
        "final_test_used": False,
    }

    selection_path.write_text(
        json.dumps(selection, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nBest eligible threshold for each model:")
    print(best_per_model.round(4).to_string(index=False))

    print("\nSelected model and threshold:")
    print(json.dumps(selection, indent=2))

    print("\nThese are pooled out-of-fold selection metrics.")
    print("They may be optimistic because they were used for selection.")
    print("The precision requirement is not guaranteed on unseen data.")
    print("The final test set remains untouched.")


if __name__ == "__main__":
    main()
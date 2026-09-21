"""Evaluate the frozen pipeline on the final holdout set."""

import hashlib
import json

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.data import PROJECT_ROOT


def main() -> None:
    """Save final metrics and charts without fitting or tuning anything."""
    metrics_dir = PROJECT_ROOT / "reports" / "metrics"
    figures_dir = PROJECT_ROOT / "reports" / "figures"
    result_path = metrics_dir / "final_test_metrics.json"

    if result_path.exists():
        raise FileExistsError(
            "Final test results already exist. Review them instead of rerunning evaluation."
        )

    model_path = PROJECT_ROOT / "models" / "churn_pipeline.joblib"
    metadata_path = PROJECT_ROOT / "models" / "model_metadata.json"
    selection_path = metrics_dir / "selected_model.json"
    train_path = PROJECT_ROOT / "data" / "processed" / "train.csv"
    test_path = PROJECT_ROOT / "data" / "processed" / "test.csv"

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    selection = json.loads(selection_path.read_text(encoding="utf-8"))

    # Verify that the saved model and its training inputs have not changed.
    checks = [
        (model_path, metadata["artifact_sha256"]),
        (train_path, metadata["training_sha256"]),
        (selection_path, metadata["selection_sha256"]),
    ]
    for path, expected_hash in checks:
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            raise ValueError(f"File changed since final fitting: {path.name}")

    artifact = joblib.load(model_path)
    pipeline = artifact["pipeline"]
    threshold = float(artifact["threshold"])

    if threshold != float(selection["threshold"]):
        raise ValueError("Saved threshold differs from the frozen selection.")
    if artifact["metadata"]["model"] != selection["model"]:
        raise ValueError("Saved model differs from the frozen selection.")

    test = pd.read_csv(test_path)
    training_ids = pd.read_csv(train_path, usecols=["customerID"])["customerID"]

    if test["customerID"].isna().any() or test["customerID"].duplicated().any():
        raise ValueError("Test customer IDs must be present and unique.")
    if set(training_ids) & set(test["customerID"]):
        raise ValueError("Training and test customers overlap.")

    X_test = test.loc[:, artifact["metadata"]["features"]]
    y_test = test["Churn"].map({"No": 0, "Yes": 1})

    if y_test.isna().any() or y_test.nunique() != 2:
        raise ValueError("Test data must contain valid Yes and No churn labels.")

    classes = pipeline.named_steps["classifier"].classes_
    positive_index = np.flatnonzero(classes == 1)
    if len(positive_index) != 1:
        raise ValueError("The model must have churn encoded as class 1.")

    probabilities = pipeline.predict_proba(X_test)[:, positive_index[0]]

    if not np.isfinite(probabilities).all():
        raise ValueError("Predictions contain non-finite probabilities.")
    if not ((probabilities >= 0) & (probabilities <= 1)).all():
        raise ValueError("Predicted probabilities must be between zero and one.")

    predictions = (probabilities >= threshold).astype(int)

    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()

    results = {
        "model": selection["model"],
        "threshold": threshold,
        "test_customers": len(test),
        "test_sha256": hashlib.sha256(test_path.read_bytes()).hexdigest(),
        "artifact_sha256": metadata["artifact_sha256"],
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }

    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["Stayed", "Churned"],
    ).plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(f"Final test confusion matrix — threshold {threshold:.2f}")
    fig.tight_layout()
    fig.savefig(figures_dir / "final_confusion_matrix.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    RocCurveDisplay.from_predictions(
        y_test,
        probabilities,
        name=selection["model"],
        ax=ax,
    )
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance ranking")
    ax.set_title("Final test ROC curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(figures_dir / "final_roc_curve.png", dpi=160)
    plt.close(fig)

    result_path.write_text(
        json.dumps(results, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(results, indent=2))
    print("\nSaved final metrics, confusion matrix, and ROC curve.")
    print("No fitting or threshold adjustment occurred during evaluation.")


if __name__ == "__main__":
    main()
"""Explain the saved model using its native feature importance."""

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data import PROJECT_ROOT
from src.preprocessing import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def main() -> None:
    """Save encoded and original-feature importance without refitting."""
    artifact = joblib.load(PROJECT_ROOT / "models" / "churn_pipeline.joblib")
    pipeline = artifact["pipeline"]

    preprocessing = pipeline.named_steps["preprocessing"]
    classifier = pipeline.named_steps["classifier"]

    feature_names = preprocessing.get_feature_names_out()
    importance = classifier.feature_importances_

    if len(feature_names) != len(importance):
        raise ValueError("Feature names and importance values do not match.")
    if not np.isfinite(importance).all():
        raise ValueError("Feature importance contains non-finite values.")

    # Our preprocessor outputs numerical fields first, then encoded categories.
    original_features = list(NUMERIC_FEATURES)
    encoder = preprocessing.named_transformers_["categorical"].named_steps["encode"]

    for feature, categories in zip(
        CATEGORICAL_FEATURES,
        encoder.categories_,
        strict=True,
    ):
        original_features.extend([feature] * len(categories))

    if len(original_features) != len(feature_names):
        raise ValueError("Original-feature mapping does not match encoded output.")

    encoded = pd.DataFrame(
        {
            "encoded_feature": feature_names,
            "original_feature": original_features,
            "importance": importance,
        }
    ).sort_values("importance", ascending=False)

    grouped = (
        encoded.groupby("original_feature", as_index=False)["importance"]
        .sum()
        .sort_values("importance", ascending=False)
    )

    metrics_dir = PROJECT_ROOT / "reports" / "metrics"
    figures_dir = PROJECT_ROOT / "reports" / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    encoded.to_csv(metrics_dir / "encoded_feature_importance.csv", index=False)
    grouped.to_csv(metrics_dir / "feature_importance.csv", index=False)

    top = grouped.head(15).sort_values("importance")

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(top["original_feature"], top["importance"], color="#2878A5")
    ax.set_xlabel("Summed native feature importance")
    ax.set_ylabel("")
    ax.set_title("Gradient Boosting: top 15 original features")
    fig.tight_layout()
    fig.savefig(figures_dir / "feature_importance.png", dpi=160)
    plt.close(fig)

    print("Importance by original feature:")
    print(grouped.round(4).to_string(index=False))

    print("\nTop 10 encoded features:")
    print(encoded.head(10).round(4).to_string(index=False))

    print("\nImportance describes contributions to training splits.")
    print("It does not establish causation or the direction of an effect.")
    print("Correlated features may share importance.")
    print("Features with more possible splits can receive greater importance.")
    print("No model fitting or final-test access occurred.")


if __name__ == "__main__":
    main()
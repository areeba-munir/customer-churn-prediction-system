"""Fit and save the locked model using training data only."""

import hashlib
import json
import platform

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.pipeline import Pipeline

from src.compare_models import build_candidates
from src.data import PROJECT_ROOT
from src.preprocessing import FEATURES, build_preprocessor


def main() -> None:
    """Train the selected pipeline and verify its saved predictions."""
    selection_path = PROJECT_ROOT / "reports" / "metrics" / "selected_model.json"
    train_path = PROJECT_ROOT / "data" / "processed" / "train.csv"
    model_dir = PROJECT_ROOT / "models"
    model_path = model_dir / "churn_pipeline.joblib"
    metadata_path = model_dir / "model_metadata.json"

    if model_path.exists() or metadata_path.exists():
        raise FileExistsError(
            "A final model or metadata file already exists. Review before replacing it."
        )

    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    candidates = build_candidates()
    model_name = selection["model"]
    threshold = float(selection["threshold"])

    if model_name not in candidates:
        raise ValueError(f"Unknown selected model: {model_name}")
    if not 0 < threshold < 1:
        raise ValueError("The decision threshold must be between zero and one.")

    data = pd.read_csv(train_path)
    X = data.loc[:, FEATURES]
    y = data["Churn"].map({"No": 0, "Yes": 1})

    if y.isna().any() or y.nunique() != 2:
        raise ValueError("Training data must contain valid Yes and No churn labels.")

    pipeline = Pipeline(
        steps=[
            ("preprocessing", build_preprocessor()),
            ("classifier", candidates[model_name]),
        ]
    )

    print(f"Fitting {model_name} on {len(X)} training customers...")
    pipeline.fit(X, y)

    metadata = {
        "model": model_name,
        "threshold": threshold,
        "positive_class": 1,
        "target_mapping": {"No": 0, "Yes": 1},
        "features": FEATURES,
        "training_rows": len(X),
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "training_sha256": hashlib.sha256(train_path.read_bytes()).hexdigest(),
        "selection_sha256": hashlib.sha256(selection_path.read_bytes()).hexdigest(),
        "classifier_parameters": pipeline.named_steps["classifier"].get_params(),
    }

    # Keep the threshold together with the complete fitted pipeline.
    artifact = {
        "pipeline": pipeline,
        "threshold": threshold,
        "metadata": metadata,
    }

    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_path)

    # Compare saved and original predictions on a small training sample.
    restored = joblib.load(model_path)
    sample = X.head(20)

    original_probabilities = pipeline.predict_proba(sample)
    restored_probabilities = restored["pipeline"].predict_proba(sample)

    np.testing.assert_allclose(
        original_probabilities,
        restored_probabilities,
        rtol=0,
        atol=1e-12,
    )

    if not np.isfinite(restored_probabilities).all():
        raise ValueError("Saved model returned non-finite probabilities.")
    if not ((restored_probabilities >= 0) & (restored_probabilities <= 1)).all():
        raise ValueError("Saved model returned probabilities outside [0, 1].")

    metadata["artifact_sha256"] = hashlib.sha256(model_path.read_bytes()).hexdigest()
    metadata_path.write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Saved model: {model_path.name}")
    print(f"Saved metadata: {metadata_path.name}")
    print(f"Decision threshold: {threshold}")
    print("Reloaded predictions match the original pipeline.")
    print("Probability range check passed.")
    print("The final test set was not loaded.")


if __name__ == "__main__":
    main()
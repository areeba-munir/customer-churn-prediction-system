# Project Requirements Checklist

Checked items are verified. Unchecked items remain pending.

## Foundation
- [x] Confirm project path and dataset choice.
- [x] Verify Python 3.12.10 and Git.
- [x] Create and verify the virtual environment.
- [x] Create local project folders.
- [x] Initialize Git on main.
- [x] Verify virtual environment and secret exclusions.
- [x] Install packages and verify imports and dependency compatibility.
- [x] Pin direct dependencies and record the complete environment.
- [x] Create and verify the initial README.
- [ ] Create and verify this checklist.
- [ ] Configure linting and testing.
- [ ] Create the first verified Git commit.

## Dataset and analysis
- [ ] Verify dataset source and usage conditions.
- [ ] Download and preserve the original raw dataset.
- [ ] Audit schema, target, missing values, duplicates, and class balance.
- [ ] Reserve a stratified final test set.
- [ ] Analyze training data and document relevant EDA findings.

## Modeling
- [ ] Build leakage-safe preprocessing.
- [ ] Train an interpretable baseline.
- [ ] Compare at least three classifiers using stratified cross-validation.
- [ ] Compare accuracy, precision, recall, F1, and ROC-AUC.
- [ ] Lock the final model and decision threshold.
- [ ] Evaluate on the final test set.
- [ ] Save actual metrics, confusion matrix, and ROC curve.
- [ ] Explain global feature importance and its limitations.

## Application and quality
- [ ] Save and reload the complete prediction pipeline.
- [ ] Build the Streamlit customer-input form.
- [ ] Validate inputs and handle prediction errors.
- [ ] Display churn status and probability.
- [ ] Verify automated tests and lint checks.
- [ ] Verify setup, training, and application startup from a fresh environment.

## Documentation and submission
- [ ] Complete README with verified commands, findings, and metrics.
- [ ] Add screenshots and known limitations.
- [ ] Publish the GitHub repository and release tag.
- [ ] Create and inspect the final source ZIP.
- [ ] Record the demonstration video.
- [ ] Upload ZIP and video and verify reviewer access.
- [ ] Submit required links and save confirmation.

## Blockers and exclusions

- No current blocker reported.
- Paid hosting is outside the current scope; the app will run locally.

# Project Requirements Checklist

Checked items have been verified.

## Foundation
- [x] Confirm project path and dataset choice.
- [x] Verify Python, Git, and virtual environment.
- [x] Create project structure and Git exclusions.
- [x] Install, pin, and verify dependencies.
- [x] Configure Ruff and Pytest.
- [x] Create milestone commits.

## Dataset and analysis
- [x] Download the IBM sample CSV and preserve the raw file.
- [x] Audit schema, target, blanks, duplicates, and class balance.
- [x] Check numerical conversion and negative numerical values.
- [x] Reserve a stratified final test set with no customer overlap.
- [x] Analyze training-set contract, tenure, internet, payment, and monthly charges.
- [ ] Complete remaining service, total-charge, and demographic exploration.
- [ ] Finish dataset attribution and reproducible download instructions.
- [ ] Visually review all generated analysis and evaluation charts.

## Modeling
- [x] Build and test leakage-safe preprocessing.
- [x] Train Logistic Regression baseline.
- [x] Compare three classifier families and a class-weighted variant.
- [x] Report required cross-validation metrics.
- [x] Select and freeze model and threshold using training data only.
- [x] Evaluate the frozen model on the final holdout.
- [x] Save final metrics, confusion matrix, and ROC curve.
- [x] Extract global feature importance with limitations.
- [x] Save and reload the complete pipeline and threshold.

## Application and testing
- [x] Validate customer inputs and service combinations.
- [x] Display predicted churn status and probability in Streamlit.
- [x] Demonstrate alert and no-alert results locally.
- [x] Verify automated Streamlit interactions.
- [x] Pass all 23 tests and Ruff checks.
- [x] Install locked dependencies in a fresh Python environment.
- [x] Pass dependency checks and all tests in the fresh environment.
- [x] Reproduce frozen-model training and evaluation in an isolated copy.
- [x] Verify identical final metrics and model artifact fingerprint.
- [x] Visually confirm the updated billing alignment.
- [ ] Save and reference application screenshots (intentionally omitted).

## Documentation and submission
- [x] Finalize README and remove resolved pending-status notes.
- [x] Document the verified isolated reproduction procedure.
- [ ] Complete final repository and ZIP content review.
- [ ] Publish GitHub repository and release tag.
- [ ] Generate the release ZIP.
- [ ] Record the demonstration video.
- [ ] Upload ZIP and video and verify reviewer access.
- [ ] Submit required links and save confirmation.

## Scope and limitations

- The application runs locally; paid hosting is not required.
- The model comparison and threshold search were not rerun during
  the isolated frozen-model reproduction check.
- Probability calibration and demographic fairness were not evaluated.
- No current execution blocker is reported.

# Reproducibility Report

## Environment

The project was developed and verified with:

* Python 3.12.10
* Dependencies pinned in `requirements-lock.txt`
* Random seed 42
* Five-fold cross-validation
* Stratified 80/20 training and final-test split

## Dataset integrity

Dataset source: [IBM Telco Customer Churn](https://github.com/IBM/telco-customer-churn-on-icp4d)

Verified dataset SHA-256:

```text
16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91
```

The source data can be downloaded and verified with:

```powershell
.\.venv\Scripts\python.exe -m src.download_data
```

The download script validates the dataset checksum and does not overwrite an existing file with different content.

## Reproduction procedure

Run the following commands in a clean clone or isolated project copy:

```powershell
Set-Location "C:\path\to\customer-churn-prediction-system"

py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt

.\.venv\Scripts\python.exe -m src.download_data
.\.venv\Scripts\python.exe -m src.data
.\.venv\Scripts\python.exe -m src.split
.\.venv\Scripts\python.exe -m src.eda
.\.venv\Scripts\python.exe -m src.train
.\.venv\Scripts\python.exe -m src.compare_models
.\.venv\Scripts\python.exe -m src.select_model
.\.venv\Scripts\python.exe -m src.fit_final
.\.venv\Scripts\python.exe -m src.evaluate
.\.venv\Scripts\python.exe -m src.explain
```

Release artifacts are overwrite-protected. A clean clone or isolated copy should be used for full reproduction instead of deleting or overwriting verified release artifacts.

The final test set must not be used to tune the model or decision threshold.

## Dataset split

The dataset was divided using a reproducible stratified split:

* Total customers: 7,043
* Training customers: 5,634
* Final-test customers: 1,409
* Training proportion: 80%
* Final-test proportion: 20%
* Random seed: 42
* Overlapping customer IDs: zero

The final test set was excluded from exploratory feature analysis, preprocessing fitting, cross-validation, model comparison, and threshold selection.

## Selected model

The selected configuration was:

* Model: Gradient Boosting
* Decision threshold: 0.23
* Positive class: churn
* Cross-validation folds: 5
* Selection rule: maximize out-of-fold recall while maintaining precision of at least 0.50

The model and threshold were selected using training-only out-of-fold predictions and frozen before final evaluation.

## Artifact verification

The independently reproduced model artifact and the main release model artifact produced the same SHA-256 value:

```text
85fb09d45130fc8b0d9debd22106440867bdf11489e6bd52db2175dc7aa82ba2
```

This verifies byte-for-byte equality of the saved prediction pipelines.

## Final holdout results

The untouched final test set contained 1,409 customers.

| Metric    | Result |
| --------- | -----: |
| Accuracy  | 73.81% |
| Precision | 50.41% |
| Recall    | 82.35% |
| F1        | 0.6254 |
| ROC-AUC   | 0.8434 |

Confusion matrix results:

| Actual outcome | Predicted stayed | Predicted churned |
| -------------- | ---------------: | ----------------: |
| Stayed         |              732 |               303 |
| Churned        |               66 |               308 |

The model identified 308 of 374 churners and missed 66. It raised 303 false churn alerts. The selected threshold prioritizes identifying churners, resulting in higher recall and more false positives.

## Verification checks

The final project verification produced:

* Dependency check: passed
* Ruff static checks: passed
* Automated tests: 23 passed
* Dataset checksum: verified
* Main model checksum: verified
* Reproduced model checksum: identical to the release model
* Streamlit application: tested locally
* High-risk prediction path: verified
* Low-risk prediction path: verified

The automated tests cover:

* Dataset loading
* Split separation and reproducibility
* Preprocessing behavior
* Customer-input validation
* Saved-model prediction
* Streamlit application interactions

## Reproducibility limitations

* Results depend on the pinned Python and package versions.
* The source dataset represents a specific historical telecom sample.
* Results may not generalize to another company, population, or time period.
* Probability calibration and demographic fairness were not evaluated.
* The selected threshold represents a recall-oriented demonstration objective rather than verified business economics.
* Generated model files should be loaded only from trusted sources because serialized Joblib files can execute code.

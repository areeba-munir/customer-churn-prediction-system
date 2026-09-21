# Customer Churn Prediction System

A machine-learning project to estimate customer churn from historical
customer data and display predictions through a Streamlit application.

## Current status

Project foundation is in progress.

Verified:
- Python 3.12.10 virtual environment.
- Git repository initialized on main.
- Local project folders created.
- Virtual environment and secrets excluded through .gitignore.
- Required packages installed and imported successfully.
- No dependency conflicts reported by pip.
- Direct dependency versions pinned.
- Complete environment recorded in requirements-lock.txt.

Dataset inspection, model training, evaluation, and the application
have not been implemented. No model results are available.

## Dataset

The IBM Telco Customer Churn dataset has been selected.
Its source and usage conditions must be verified before downloading.
The original raw dataset will remain unchanged.
Churn will be the positive class after the target is inspected.

## Local setup

Prerequisites: Windows, PowerShell, Git, and Python 3.12.
Adjust the path when setting up on another computer.

Run these commands in PowerShell, in order:

    Set-Location "C:\Mydata\Interns\Enliven solutions LLC\project 6\customer-churn-prediction-system"
    py -3.12 -m venv .venv
    .\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
    .\.venv\Scripts\python.exe -m pip check

Installation from requirements-lock.txt still needs verification in a
fresh environment. The current environment was installed using
requirements-dev.txt and then recorded in requirements-lock.txt.

## Project organization

- app/: Streamlit application.
- data/raw/: Original dataset.
- data/processed/: Generated intermediate data, excluded from Git.
- models/: Saved preprocessing and model pipeline.
- notebooks/: Exploratory analysis.
- reports/figures/: Analysis and evaluation charts.
- reports/metrics/: Model comparison and evaluation results.
- reports/screenshots/: Application screenshots.
- src/: Reusable Python modules.
- tests/: Automated tests.

Empty folders will appear in Git once they contain tracked files.

## Dependency files

- requirements.txt: Pinned application dependencies.
- requirements-dev.txt: Application dependencies and development tools.
- requirements-lock.txt: Complete development environment snapshot.

## Planned workflow

1. Verify the dataset source and inspect the raw data.
2. Reserve a final test set before exploratory model decisions.
3. Analyze training data and build reproducible preprocessing.
4. Compare at least three classifiers using training data.
5. Lock the final model and decision threshold.
6. Evaluate on the final test set and explain the model.
7. Save the pipeline and connect it to Streamlit.
8. Complete tests, documentation, and submission materials.

Training, evaluation, and application commands will be added after
implementation and verification.

## Limitations

No trained model or prediction application is available yet.
Predictions will be estimates, not guaranteed outcomes.
Feature associations and importance do not establish causation.

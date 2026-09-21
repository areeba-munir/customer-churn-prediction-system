# Model Selection Decision

## Selected configuration

- Model: Gradient Boosting.
- Decision threshold: 0.23.
- Predict churn when predicted churn probability is >= 0.23.
- Model parameters: Gradient Boosting configuration in src/compare_models.py.
- Random seed: 42.

## Selection method

Compared Logistic Regression, balanced Logistic Regression, Random Forest,
and Gradient Boosting using the same five stratified training-data folds.

Preprocessing was fitted independently inside each training fold.
Thresholds from 0.10 through 0.90, in increments of 0.01, were compared
using pooled out-of-fold predictions.

The selection rule maximized recall subject to precision of at least 0.50.
Equal-recall candidates were ranked by precision, then F1.

The precision requirement is a demonstration assumption.
Actual outreach costs and retention benefits were unavailable.

## Selected validation results

- Accuracy: 0.7387291444799432.
- Precision: 0.5047150471504716.
- Recall: 0.8234113712374582.
- F1: 0.6258261311642095.
- ROC-AUC: 0.84714902942158.

Gradient Boosting and ordinary Logistic Regression achieved the same
recall at their selected thresholds. Gradient Boosting had higher precision.

## Limitations

These metrics were used for model and threshold selection and may be
optimistic. They are not final holdout-test results.

The precision requirement is not guaranteed on unseen customers.
Approximately half of validation alerts were false positives.

## Final evaluation rule

Freeze this model configuration and threshold before evaluating the
final test set. Fit the selected pipeline using all training customers.

Do not adjust the model or threshold based on final test results.
Report the holdout results honestly, including any performance decline.

The final test set has not been used at this selection milestone.
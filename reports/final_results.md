# Final Evaluation and Explainability

## Frozen configuration

- Model: Gradient Boosting.
- Training customers: 5,634.
- Holdout customers: 1,409.
- Churn threshold: 0.23.
- Positive class: Yes, encoded as 1.

The model and threshold were selected using training-data cross-validation
and frozen before holdout evaluation.

## Holdout results

| Metric | Result |
|---|---:|
| Accuracy | 73.81% |
| Precision | 50.41% |
| Recall | 82.35% |
| F1 | 0.6254 |
| ROC-AUC | 0.8434 |

| Outcome | Customers |
|---|---:|
| True negatives | 732 |
| False positives | 303 |
| False negatives | 66 |
| True positives | 308 |

The model identified 308 of 374 actual churners and missed 66.
Of 611 churn alerts, 308 were correct and 303 were false alarms.

The lower threshold prioritizes recall. Overall accuracy is close to
the majority-class baseline, so accuracy alone does not communicate
the model's usefulness for detecting churn.

The demonstration precision criterion of at least 50% was met on this
holdout set. It is not guaranteed on future data. Business profitability
cannot be established without outreach costs and intervention outcomes.

## Global explainability

Native Gradient Boosting importance, summed by original input field:

- Contract: approximately 38.92%.
- Tenure: approximately 13.82%.
- Internet service: approximately 8.69%.
- Total charges: approximately 8.66%.
- Monthly charges: approximately 8.18%.

The month-to-month contract indicator was the largest individual
encoded feature by native importance.

These values describe contributions to training splits. They do not
establish causation, indicate effect direction, or explain an individual
prediction. Correlated features can share importance, and features with
more possible splits can receive greater importance.

## Artifact verification

- Saved preprocessing and classifier together with the decision threshold.
- Reloaded predictions matched the original fitted pipeline.
- Prediction probabilities passed finite-value and range checks.
- Loading and prediction succeeded in a fresh Python process.
- Final evaluation did not fit or tune the model.

## Remaining checks

- Visual review of generated charts.
- Prediction-input validation and automated tests.
- Streamlit interface and application tests.
- Full documentation, reproducibility checks, and submission materials.
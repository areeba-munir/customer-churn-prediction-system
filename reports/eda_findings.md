# Initial Exploratory Analysis

## Scope

Analysis used only the 5,634 training customers.
The training churn rate was 26.54%.
The final test set was excluded from exploratory analysis.

## Findings

- Month-to-month customers had a 42.75% churn rate, compared with
  11.08% for one-year and 2.87% for two-year contracts.
- Customers with 0-12 months of tenure had a 47.32% churn rate,
  compared with 9.95% for customers with 49+ months of tenure.
- Fiber-optic customers had a 42.09% churn rate, compared with
  18.69% for DSL and 7.25% for customers without internet service.
- Electronic-check customers had a 45.74% churn rate. Other payment
  groups ranged from 14.92% to 19.28%.
- Median monthly charges were 79.95 among customers who churned
  and 64.40 among customers who stayed.

## Implications

Contract type, tenure, services, payment method, and charges are
candidates for modeling. These univariate comparisons do not measure
independent effects or establish a feature-importance ranking.

The patterns suggest groups worth investigating for retention outreach.
They do not establish that changing a contract, service, or payment
method would prevent churn.

## Additional focused findings

- Customers without online security had a 41.94% churn rate, compared
  with 14.42% for customers with online security.
- Customers without technical support had a 41.75% churn rate, compared
  with 15.16% for customers with technical support.
- Senior citizens had a 41.09% churn rate, compared with 23.70% for
  non-senior customers.
- Median monthly charges were 79.95 for churners and 64.40 for
  non-churners.
- Median total charges were 740.30 for churners and 1,691.90 for
  non-churners.
- The generated EDA charts were visually reviewed for readability,
  labeling, and consistency with the saved metrics.

## Limitations

- Findings describe this sample and may not generalize to other companies.
- Features may be related to one another.
- These results show associations and do not establish causation.
- The training data contained eight missing `TotalCharges` values among
  customers who did not churn.
- Demographic fairness and probability calibration were not evaluated.

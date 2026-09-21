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

## Limitations and pending analysis

- Findings describe this sample and may not generalize to other companies.
- Features may be related to one another.
- Total charges, additional service fields, and demographic fields
  still need focused exploration.
- Generated charts still need visual review.
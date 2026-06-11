# 50 Startups Profit Prediction: Detailed Project Log

## Project Overview

This project uses the Kaggle-style `50_Startups.csv` dataset to predict startup
`Profit`. It is a supervised machine-learning regression problem implemented
with the CRISP-DM framework.

The dataset contains 50 observations and five columns:

| Column | Meaning | Role |
|---|---|---|
| `R&D Spend` | Research and development spending | Predictor |
| `Administration` | Administration spending | Predictor |
| `Marketing Spend` | Marketing spending | Predictor |
| `State` | California, Florida, or New York | Predictor |
| `Profit` | Startup profit | Target |

The attached reference text discussed regression, EDA, categorical encoding,
feature selection, Linear Regression, Random Forest, XGBoost, evaluation, and
CRISP-DM. Those relevant concepts were incorporated into the implemented
workflow.

## CRISP-DM 1: Business Understanding

### Objective

Predict startup profit using spending and state information.

### Business Questions

- Which spending category is most strongly associated with profit?
- Does `Administration` provide useful predictive information?
- Does startup location meaningfully affect profit?
- Which model provides the most reliable predictions?

### Important Limitation

The analysis identifies predictive relationships, not causal effects. For
example, a high R&D coefficient does not prove that increasing R&D spending
will always cause profit to increase.

## CRISP-DM 2: Data Understanding and EDA

### Data Quality

- Rows: 50
- Columns: 5
- Missing values: 0
- Duplicate rows: 0
- California observations: 17
- Florida observations: 16
- New York observations: 17

### Numeric Summary

| Variable | Minimum | Mean | Maximum |
|---|---:|---:|---:|
| `R&D Spend` | 0.00 | 73,721.62 | 165,349.20 |
| `Administration` | 51,283.14 | 121,344.64 | 182,645.56 |
| `Marketing Spend` | 0.00 | 211,025.10 | 471,784.10 |
| `Profit` | 14,681.40 | 112,012.64 | 192,261.83 |

### Correlation with Profit

| Feature | Correlation |
|---|---:|
| `R&D Spend` | 0.9729 |
| `Marketing Spend` | 0.7478 |
| `Administration` | 0.2007 |

`R&D Spend` has the strongest linear relationship with profit.
`Administration` has a weak marginal relationship, but this alone is not a
sufficient reason to remove it.

### Zero Values and Outliers

- `R&D Spend` contains 2 zero values.
- `Marketing Spend` contains 3 zero values.
- `Administration` contains no zero values.
- `Profit` contains 1 potential IQR outlier.

Zero spending values were retained because they can legitimately mean that a
startup did not spend money in that category. The potential profit outlier was
flagged but retained because removing observations from a dataset of only 50
rows could distort the analysis.

### Multicollinearity Findings

The engineered features were checked against the original spending variables:

| Feature Pair | Correlation |
|---|---:|
| `Marketing Spend` and `Total Spend` | 0.9521 |
| `R&D Spend` and `Total Spend` | 0.8697 |
| `R&D Spend` and `R&D Ratio` | 0.8002 |

Because `Total Spend` is constructed from the original spending variables,
using all of them together would introduce avoidable multicollinearity.

### EDA Artifacts

- `artifacts/eda_distributions.png`
- `artifacts/eda_rd_spend_vs_profit.png`
- `artifacts/eda_boxplots.png`
- `artifacts/eda_correlation_matrix.csv`
- `artifacts/eda_high_correlations.csv`
- `artifacts/eda_outlier_flags.csv`

## CRISP-DM 3: Data Preparation

### Cleaning and Transformation Pipeline

- Numeric missing values are imputed with the median.
- Categorical missing values are imputed with the most frequent category.
- Numeric features are standardized.
- `State` is one-hot encoded with one reference category dropped.
- Unknown state categories can be handled by the model pipeline.
- Feature engineering creates `Total Spend` and `R&D Ratio` for comparison.

The attachment described several encoding methods. One-hot encoding was chosen
because `State` is nominal, has only three categories, and one-hot encoding does
not create an artificial order such as California < Florida < New York.

### Feature-Set Comparison

Feature sets were compared using only training data and shuffled five-fold
cross-validation:

| Feature Set | CV R2 | CV RMSE |
|---|---:|---:|
| Original features | 0.9340 | **9,550.68** |
| Without `Administration` | 0.9364 | 9,655.47 |
| `Total Spend`, `R&D Ratio`, and `State` | 0.8885 | 12,551.75 |

The original features were selected because they produced the lowest CV RMSE:

- `R&D Spend`
- `Administration`
- `Marketing Spend`
- `State`

Although removing `Administration` produced a slightly higher CV R2, it
produced a worse CV RMSE. Because RMSE is the chosen selection metric,
`Administration` was retained.

### Top Five Feature-Selection Methods

The attached text requested these five feature-selection approaches:

1. Correlation Analysis: fast filter method for linear relationships.
2. Mutual Information: filter method that can identify nonlinear dependence.
3. Recursive Feature Elimination: wrapper method using repeated model fitting.
4. Lasso: embedded method that can reduce weak coefficients to zero.
5. Random Forest Feature Importance: embedded tree-based importance method.

All five methods were applied using training data only to avoid test-data
leakage.

| Overall Rank | Feature | Average Rank |
|---:|---|---:|
| 1 | `R&D Spend` | 1.0 |
| 2 | `Marketing Spend` | 2.0 |
| 3 | `Administration` | 3.4 |
| 4 | `State_New York` | 3.8 |
| 5 | `State_Florida` | 4.0 |

Detailed results are saved in
`artifacts/feature_selection_results.csv`.

## Feature Interpretation

### Why Administration Is Weak

Administration spending is usually an operating cost rather than a direct
revenue-generating investment. After controlling for R&D and marketing,
administration contributes little additional predictive information.

It should not be removed solely because of its low correlation. The
cross-validation comparison showed that retaining it slightly improved RMSE.

### State Interpretation

California is the reference category created by one-hot encoding. It is not
selected because it is better than other states.

After controlling for spending, the full-data Linear Regression state
coefficients were approximately:

| State | Predicted Difference vs California |
|---|---:|
| California | 0.00 |
| Florida | +198.79 |
| New York | -41.89 |

These differences are extremely small compared with average profit. Therefore,
the dataset provides little evidence that state adds meaningful predictive
value after spending is controlled.

Raw average profit also does not show that California is better:

| State | Average Profit |
|---|---:|
| Florida | 118,774.02 |
| New York | 113,756.45 |
| California | 103,905.18 |

Raw averages may reflect different spending levels and must not be interpreted
as causal state effects.

## CRISP-DM 4: Modeling

Three candidate regression models were trained using consistent preprocessing:

- Multiple Linear Regression
- Random Forest Regressor
- XGBoost Regressor

The dataset was split into 80% training data and 20% test data with
`random_state=42`. Shuffled five-fold cross-validation was used because the CSV
rows are approximately ordered by profit; non-shuffled folds would produce a
misleading evaluation.

## CRISP-DM 5: Evaluation

### Metrics

- R2 measures the proportion of target variance explained by the model.
- RMSE measures typical prediction error in profit units and penalizes larger
  errors more strongly.

### Results

| Model | CV R2 | CV RMSE | Test R2 | Test RMSE |
|---|---:|---:|---:|---:|
| Linear Regression | 0.9340 | **9,550.68** | 0.8987 | 9,055.96 |
| Random Forest | 0.9360 | 9,667.17 | **0.9080** | **8,629.20** |
| XGBoost | 0.9005 | 11,625.48 | 0.8025 | 12,647.34 |

Random Forest performed best on the single holdout test set. Linear Regression
had the lowest cross-validation RMSE and was selected because model selection
should not be based on the test set.

The attached text included example performance ranges such as very high
XGBoost R2 values. Those ranges are not guaranteed. In the implemented
experiment, XGBoost performed worse, likely because the dataset is very small
and the relationship is already mostly linear.

## CRISP-DM 6: Deployment

The selected Linear Regression pipeline is retrained on all available data and
saved as:

```text
artifacts/best_model.joblib
```

The FastAPI application in `app.py` provides:

- `GET /`: basic usage message
- `GET /health`: model availability check
- `POST /predict`: profit prediction
- `/docs`: interactive API documentation

Run the workflow:

```powershell
python .\train_linear_regression.py
```

Start the API:

```powershell
uvicorn app:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

The API was verified to load the saved pipeline and return predictions.

## Final Conclusion

`R&D Spend` is consistently the strongest profit predictor across correlation,
Mutual Information, RFE, Lasso, and Random Forest importance. Marketing spend
is the second most useful predictor. Administration and state provide limited
but potentially useful additional information.

Linear Regression is the preferred deployment model because it achieved the
lowest cross-validation RMSE, is stable for this small dataset, and is easy to
interpret. Results should be treated cautiously because the dataset contains
only 50 observations and lacks potentially important variables such as
industry, company age, funding, tax rates, and employee count.

## Project Files

| File | Purpose |
|---|---|
| `50_Startups.csv` | Source data |
| `train_linear_regression.py` | CRISP-DM training and evaluation pipeline |
| `model_utils.py` | Reusable feature-engineering transformation |
| `app.py` | FastAPI deployment application |
| `requirements.txt` | Python dependencies |
| `artifacts/model_results.csv` | Model comparison |
| `artifacts/feature_set_results.csv` | Feature-set comparison |
| `artifacts/feature_selection_results.csv` | Five-method feature ranking |
| `artifacts/model_metadata.json` | Selected model and feature metadata |

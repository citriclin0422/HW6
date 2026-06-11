# HW6: Startup Profit Prediction and Feature Selection

## Project Summary

This project predicts startup profit using the `50_Startups.csv` dataset and
follows the CRISP-DM machine-learning process. The final workflow performs
exploratory data analysis, preprocessing, feature selection, Linear Regression
training, evaluation, visualization, model serialization, and API deployment.

The dataset contains 50 startups and the following variables:

| Variable | Type | Role |
|---|---|---|
| `R&D Spend` | Numeric | Predictor |
| `Administration` | Numeric | Predictor |
| `Marketing Spend` | Numeric | Predictor |
| `State` | Categorical | Predictor |
| `Profit` | Numeric | Target |

### Project Workflow

![Startup profit prediction workflow](artifacts/workflow.png)

## 1. Business Understanding

The objective is to predict startup profit from spending patterns and company
location. The analysis also investigates which inputs contribute useful
predictive information.

Main business questions:

- Is R&D spending the strongest predictor of profit?
- Does marketing spending improve prediction?
- Does administration spending provide useful information?
- Does startup location meaningfully affect profit?

The model identifies predictive relationships, not causal effects. A strong
relationship between R&D spending and profit does not prove that increasing R&D
spending will always cause profit to increase.

## 2. Data Understanding

The dataset contains 50 rows and 5 columns. No missing values or duplicate rows
were found.

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

`R&D Spend` has the strongest linear relationship with profit. Marketing has a
moderate positive relationship, while Administration has a weak relationship.

Zero spending values and the potential Profit outlier were retained because
they may represent valid observations, and removing records from a 50-row
dataset could distort the analysis.

## 3. Data Preparation

The preprocessing and modeling workflow uses a Scikit-Learn `Pipeline`:

- Numeric features are standardized with `StandardScaler`.
- `State` is one-hot encoded with the first category dropped.
- Unknown state categories are handled safely.
- Data is split into 80% training and 20% testing.
- `random_state=42` is used for reproducibility.

Dropping the first state category avoids the dummy-variable trap. California is
therefore treated as the reference category in the deployed model.

## 4. Feature Selection

Five feature-selection algorithms were implemented:

1. **Correlation Analysis** ranks features by absolute Pearson correlation.
2. **SelectKBest F-Regression** ranks features using individual F-statistics.
3. **Recursive Feature Elimination (RFE)** repeatedly removes weak features.
4. **Lasso Regression** ranks features using absolute L1-regularized coefficients.
5. **Random Forest Importance** ranks features using tree-based importance.

The selectors are fitted using training data only. Their top 1-5 features are
then evaluated with the same Linear Regression model on the held-out test set.

### Feature-Selection Scores

| Feature | Correlation | F-Score | RFE Rank | Lasso Coefficient | Random Forest Importance |
|---|---:|---:|---:|---:|---:|
| `R&D Spend` | 0.9730 | 676.1035 | 1 | 37,947.57 | 0.9273 |
| `Marketing Spend` | 0.7738 | 56.6907 | 2 | 3,457.12 | 0.0629 |
| `Administration` | 0.0902 | 0.3120 | 3 | -1,736.60 | 0.0069 |
| `State_Florida` | 0.0955 | 0.3497 | 4 | 493.91 | 0.0017 |
| `State_New York` | 0.1050 | 0.4238 | 5 | 0.00 | 0.0012 |

All five algorithms selected `R&D Spend` as the strongest feature and
`Marketing Spend` as the second strongest.

### Test-Set Feature-Selection Performance

| Selected Feature Set | Test RMSE | Test R-squared |
|---|---:|---:|
| `R&D Spend` | **7,714.33** | **0.9265** |
| `R&D Spend`, `Marketing Spend` | 8,206.33 | 0.9168 |
| Correlation/F-Regression top 3 | 8,242.78 | 0.9161 |
| RFE/Lasso/Random Forest top 3 | 8,995.91 | 0.9001 |
| All deployed-model features | 9,055.96 | 0.8987 |

On the fixed holdout test set, the best result uses only `R&D Spend`. However,
the dataset is very small, so this result may depend strongly on the particular
train/test split.

![Feature-selection performance comparison](artifacts/feature_selection_performance_allinone.png)

## 5. Cross-Validation Feature-Count Comparison

A separate shuffled five-fold cross-validation experiment compares cumulative
expert-ranked features across the full dataset.

| Feature Count | Selected Features | CV RMSE | CV R-squared |
|---:|---|---:|---:|
| 1 | `R&D Spend` | 9,091.21 | 0.9374 |
| 2 | `R&D Spend`, `Marketing Spend` | **8,883.70** | **0.9389** |
| 3 | Add `State_New York` | 9,226.66 | 0.9342 |
| 4 | Add `State_Florida` | 9,249.63 | 0.9339 |
| 5 | Add `State_California` | 9,249.63 | 0.9339 |

This experiment indicates that `R&D Spend` plus `Marketing Spend` provides the
best average cross-validation performance. Adding state features reduces
performance.

![Feature-count comparison](artifacts/final_output_diagram.gif)

The holdout and cross-validation experiments answer different questions. The
single holdout split favors one feature, while repeated folds favor two
features. Because cross-validation is generally more stable for small datasets,
the two-feature result is stronger evidence for feature-count selection.

## 6. Modeling and Evaluation

The deployed baseline is Multiple Linear Regression. It uses all original
inputs after preprocessing.

| Metric | Result |
|---|---:|
| Test RMSE | 9,055.96 |
| Test R-squared | 0.8987 |
| Training rows | 40 |
| Test rows | 10 |

The full Linear Regression model explains approximately 89.87% of the Profit
variance on the held-out test set.

Earlier model comparisons produced the following results:

| Model | CV R-squared | CV RMSE | Test R-squared | Test RMSE |
|---|---:|---:|---:|---:|
| Linear Regression | 0.9340 | **9,550.68** | 0.8987 | 9,055.96 |
| Random Forest | 0.9360 | 9,667.17 | **0.9080** | **8,629.20** |
| XGBoost | 0.9005 | 11,625.48 | 0.8025 | 12,647.34 |

Random Forest performs best on the single test split, but Linear Regression has
the lowest cross-validation RMSE. Linear Regression remains the deployment
model because it is stable, interpretable, and appropriate for this small,
mostly linear dataset.

## 7. Deployment

The fitted preprocessing and Linear Regression pipeline is saved as:

- `startup_profit_model.pkl`
- `artifacts/best_model.joblib`

The FastAPI application in `app.py` provides:

- `GET /`
- `GET /health`
- `POST /predict`
- Interactive API documentation at `/docs`

Run the complete workflow:

```powershell
python .\train_linear_regression.py
```

Start the API:

```powershell
uvicorn app:app --reload
```

## Final Conclusions

- `R&D Spend` is consistently the strongest predictor of startup profit.
- `Marketing Spend` is consistently the second most useful feature.
- Administration and state provide limited additional predictive value.
- The best holdout result uses only `R&D Spend`.
- The more stable cross-validation comparison favors `R&D Spend` and
  `Marketing Spend`.
- Linear Regression is suitable for deployment because it is interpretable and
  performs reliably on this small dataset.
- Results should be interpreted cautiously because only 50 observations are
  available and important business variables may be missing.

## Main Project Artifacts

| Artifact | Purpose |
|---|---|
| `train_linear_regression.py` | Complete CRISP-DM training workflow |
| `artifacts/workflow.png` | Complete CRISP-DM workflow diagram |
| `artifacts/feature_selection_results.csv` | Five-algorithm feature scores |
| `artifacts/feature_selection_performance_allinone.csv` | Feature-count 1-5 test results |
| `artifacts/feature_selection_performance_allinone.png` | All-in-one performance figure |
| `artifacts/feature_count_comparison.csv` | Cross-validation feature-count results |
| `artifacts/final_output_diagram.gif` | Reference-style feature-count figure |
| `artifacts/model_metadata.json` | Final model evaluation metadata |
| `artifacts/best_model.joblib` | Deployable fitted pipeline |
| `app.py` | FastAPI prediction service |

# 50 Startups: CRISP-DM Machine Learning Project Log

**Date:** June 9, 2026
**Objective:** Build a robust, self-documenting Machine Learning pipeline to predict startup profitability based on spending profiles, adhering strictly to the CRISP-DM methodology.

---

## 1. Data Acquisition & Sourcing
*   **Action:** Sourced the classic `50 Startups` dataset from an open-source GitHub repository after verifying it contained exactly 50 records.
*   **Result:** Saved the dataset locally as `data.csv`. 
*   **Dataset Features:** `R&D Spend`, `Administration`, `Marketing Spend`, `State` (Categorical), and `Profit` (Target Variable).

## 2. CRISP-DM Step 1: Business Understanding
*   **Objective Defined:** Predict the net profit of a startup based on its historical expenditures and geographical location. This predictive capability allows venture capitalists to dynamically evaluate and identify which startups possess the highest mathematical potential for profitability.

## 3. CRISP-DM Step 2: Data Understanding (EDA)
*   **Action:** Integrated a massive Exploratory Data Analysis (EDA) suite directly into the core Python script (`solve_50_startups.py`) using `pandas`, `matplotlib`, and `seaborn`.
*   **Checks Performed:** 
    *   Dataset shape (50, 5), Data Types, and Info.
    *   Missing Values (None found).
    *   Duplicate Records (None found).
    *   Descriptive statistics to understand mean and standard deviations of spending.
*   **Visualizations Generated:**
    *   Histograms to verify normal/skewed distributions.
    *   Boxplots to detect extreme outliers (minimal found, none removed).
    *   Correlation Matrix Heatmap to analyze linear relationships.
    *   Scatter plots to visualize `Profit` vs individual expenditures.
    *   Categorical plots (Countplots, Boxplots) to assess Profit across `California`, `Florida`, and `New York`.
*   **Key EDA Findings:** `R&D Spend` possesses an incredibly high correlation (0.97) with Profit. `Administration` possesses a very weak correlation (0.20).

## 4. CRISP-DM Step 3: Data Preparation
*   **Action:** Constructed a robust, production-ready preprocessing pipeline to prevent data leakage and handle varied data types.
*   **Categorical Handling:** Applied `OneHotEncoder` to the `State` column. Used `drop='first'` to drop the baseline state, successfully preventing the "dummy variable trap" (multicollinearity).
*   **Numerical Handling:** Applied `StandardScaler` to `R&D Spend`, `Administration`, and `Marketing Spend` to center the mean at 0 and scale the variance to 1.
*   **Data Splitting:** Applied an 80/20 Train-Test split (`train_test_split`) using a fixed random state for reproducibility.

## 5. CRISP-DM Step 3.11: Advanced Feature Selection Analysis
*   **Action:** Implemented a mid-pipeline analytical suite consisting of 5 independent Feature Selection methodologies to statistically determine if weak features (`Administration` and `State`) should be dropped.
*   **Methods Applied & Results:**
    1.  **Pearson Correlation:** Confirmed `R&D Spend` dominated, while `Administration` and `State` scored < 0.11 on transformed data.
    2.  **SelectKBest (F-Regression):** F-Score for `R&D Spend` was 676.10. `Administration` scored only 0.31.
    3.  **Recursive Feature Elimination (RFE):** Ranked `R&D Spend` as #1 and `Marketing Spend` as #2.
    4.  **Lasso Regularization (L1 Penalty):** Retained a massive positive coefficient for `R&D Spend` but yielded a negative coefficient for `Administration`.
    5.  **Random Forest Importance:** Showed `R&D Spend` accounted for ~92.2% of tree variance logic.
*   **Conclusion:** Features like `Administration` and `State` offer mathematically negligible predictive value. While retained for the baseline model, they are prime targets for removal in a hyper-optimized pipeline.

## 6. CRISP-DM Step 4 & 5: Modeling & Evaluation
*   **Action:** Passed the preprocessed 80% training data into a `Multiple Linear Regression` algorithm via a Scikit-Learn `Pipeline`.
*   **Evaluation on 20% Test Set:**
    *   **Root Mean Squared Error (RMSE):** `$9,055.96`
    *   **R-Squared (R2) Score:** `0.8987`
*   **Result:** The model successfully explains almost 90% of the variance in startup profit, a highly accurate result for a baseline linear model.

## 7. CRISP-DM Step 6: Deployment
*   **Action:** Serialized the entire, fitted Scikit-Learn `Pipeline` (which bundles the One-Hot Encoder, the Standard Scaler, and the Linear Regression weights).
*   **Result:** Exported as `startup_profit_model.pkl` using the `joblib` library. The project is completely decoupled and ready to be loaded into a REST API (like FastAPI or Flask) for live venture capital prediction environments.

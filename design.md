project:
  title: "Startup Profit Prediction"
  dataset: "Kaggle 50 Startups Dataset"
  methodology: "CRISP-DM"
  problem_type: "Supervised Learning - Regression"
  target_variable: "Profit"

objective:
  description: >
    Build a machine learning model to predict startup profit
    based on spending patterns and company location.
  key_question: >
    Can startup profit be predicted using R&D Spend,
    Administration, Marketing Spend, and State?

features:
  - name: "R&D Spend"
    type: "Numerical"
    business_meaning: "Investment in research, product development, and innovation."
    expected_relationship: "Positive relationship with Profit"
    importance: "Very High"
    expert_note: >
      Usually the strongest predictor because innovation and product
      development directly influence long-term business growth.

  - name: "Administration"
    type: "Numerical"
    business_meaning: "Cost of management, office operations, HR, legal, and accounting."
    expected_relationship: "Weak or unclear relationship with Profit"
    importance: "Low to Medium"
    expert_note: >
      Administration supports operations but does not directly generate revenue.
      Excessive administration cost may reduce profitability.

  - name: "Marketing Spend"
    type: "Numerical"
    business_meaning: "Spending on advertising, branding, campaigns, and customer acquisition."
    expected_relationship: "Positive relationship with Profit"
    importance: "Medium to High"
    expert_note: >
      Marketing can increase sales and customer reach, but its effectiveness
      depends on campaign quality and market response.

  - name: "State"
    type: "Categorical"
    business_meaning: "The location where the startup operates."
    possible_values:
      - "California"
      - "Florida"
      - "New York"
    expected_relationship: "Indirect relationship with Profit"
    importance: "Low"
    preprocessing: "One-Hot Encoding"
    expert_note: >
      State may reflect tax policy, talent availability, investor access,
      and operating cost, but it is usually less important than spending variables.

crisp_dm:
  step_1_business_understanding:
    goal: "Define the business problem and prediction objective."
    business_problem: >
      Startups have limited budgets and need to decide how to allocate
      resources effectively.
    questions:
      - "Does R&D spending increase profit?"
      - "Does marketing spending improve business performance?"
      - "Is administration spending useful or only a cost?"
      - "Does company location influence profit?"
    expected_output:
      - "Clear business objective"
      - "Regression problem definition"
      - "Target variable identified"

  step_2_data_understanding:
    goal: "Explore the dataset and understand feature relationships."
    tasks:
      - "Load the dataset"
      - "Check column names"
      - "Check data types"
      - "Check missing values"
      - "Review summary statistics"
      - "Analyze correlation between spending variables and Profit"
      - "Visualize relationships using scatter plots"
    expected_findings:
      - "R&D Spend usually has the strongest relationship with Profit"
      - "Marketing Spend may have moderate positive relationship with Profit"
      - "Administration may have weak relationship with Profit"
      - "State needs categorical encoding"

  step_3_data_preparation:
    goal: "Prepare data for machine learning modeling."
    tasks:
      - "Separate input features X and target y"
      - "Encode State using One-Hot Encoding"
      - "Split dataset into training and testing sets"
      - "Check final feature columns"
    input_features:
      - "R&D Spend"
      - "Administration"
      - "Marketing Spend"
      - "State"
    target:
      - "Profit"
    encoding:
      method: "One-Hot Encoding"
      column: "State"
      drop_first: true
      reason: "Avoid dummy variable trap"
    train_test_split:
      test_size: 0.2
      random_state: 42

  step_4_modeling:
    goal: "Train regression models to predict Profit."
    baseline_model:
      name: "Linear Regression"
      reason: >
        Linear Regression is simple, interpretable, and suitable
        for understanding how spending variables affect Profit.
    formula: >
      Profit = β0
             + β1 * R&D Spend
             + β2 * Administration
             + β3 * Marketing Spend
             + β4 * State_Florida
             + β5 * State_New York
    optional_models:
      - name: "Ridge Regression"
        purpose: "Reduce overfitting using L2 regularization"
      - name: "Lasso Regression"
        purpose: "Feature selection using L1 regularization"
      - name: "Decision Tree Regressor"
        purpose: "Capture nonlinear relationships"
      - name: "Random Forest Regressor"
        purpose: "Improve prediction stability using ensemble learning"

  step_5_evaluation:
    goal: "Measure model performance."
    metrics:
      - name: "MAE"
        full_name: "Mean Absolute Error"
        meaning: "Average absolute prediction error"
        interpretation: "Lower is better"
      - name: "MSE"
        full_name: "Mean Squared Error"
        meaning: "Average squared prediction error"
        interpretation: "Lower is better"
      - name: "RMSE"
        full_name: "Root Mean Squared Error"
        meaning: "Prediction error in the same unit as Profit"
        interpretation: "Lower is better"
      - name: "R2 Score"
        full_name: "Coefficient of Determination"
        meaning: "Percentage of Profit variance explained by the model"
        interpretation: "Closer to 1 is better"
    caution: >
      Because the dataset contains only 50 rows, evaluation results may change
      depending on train-test split. Results should be treated as a learning
      example, not a production-grade prediction system.

  step_6_deployment:
    goal: "Make the model useful for business decision support."
    deployment_type:
      - "Report"
      - "Dashboard"
      - "Web application"
      - "API service"
      - "Streamlit prototype"
      - "FastAPI backend"
    user_flow:
      - "User inputs startup spending data"
      - "System applies preprocessing"
      - "Model predicts Profit"
      - "System displays predicted Profit"
      - "User uses result for decision making"
    example_input:
      rd_spend: 100000
      administration: 120000
      marketing_spend: 250000
      state: "New York"
    example_output:
      predicted_profit: 145000

python_workflow:
  libraries:
    - "pandas"
    - "numpy"
    - "scikit-learn"

  code: |
    import pandas as pd
    import numpy as np

    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    # Load dataset
    df = pd.read_csv("50_Startups.csv")

    # Data understanding
    print(df.head())
    print(df.info())
    print(df.describe())
    print(df.isnull().sum())

    # Separate features and target
    X = df[["R&D Spend", "Administration", "Marketing Spend", "State"]]
    y = df["Profit"]

    # One-Hot Encoding
    X = pd.get_dummies(X, columns=["State"], drop_first=True)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Evaluation
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print("MAE:", mae)
    print("MSE:", mse)
    print("RMSE:", rmse)
    print("R2 Score:", r2)

    # Coefficient analysis
    coef_df = pd.DataFrame({
        "Feature": X.columns,
        "Coefficient": model.coef_
    })

    print(coef_df)
    print("Intercept:", model.intercept_)

expert_summary:
  feature_importance_ranking:
    - rank: 1
      feature: "R&D Spend"
      reason: "Main driver of innovation and profit growth"
    - rank: 2
      feature: "Marketing Spend"
      reason: "Supports customer acquisition and sales growth"
    - rank: 3
      feature: "Administration"
      reason: "Supports operations but may not directly increase profit"
    - rank: 4
      feature: "State"
      reason: "Location factor with indirect influence"

  key_insight: >
    Startup profit is usually driven more by productive investment,
    especially R&D Spend, than by general administration cost.

  business_recommendations:
    - "Prioritize efficient R&D investment"
    - "Optimize marketing budget based on return on investment"
    - "Control administration costs"
    - "Use State as a supporting factor, not the main decision factor"
    - "Use the model as a decision-support tool, not as the only decision maker"
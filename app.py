"""FastAPI deployment and interactive dashboard for the startup profit models."""

from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from train_linear_regression import (
    RAW_FEATURES,
    TARGET,
    analyze_feature_selection,
    build_feature_rankings,
    build_pipeline,
)


BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / "artifacts" / "best_model.joblib"
DATA_FILE = BASE_DIR / "50_Startups.csv"
DASHBOARD_FILE = BASE_DIR / "dashboard.html"
STATES = {"California", "Florida", "New York"}
ALGORITHMS = [
    "Correlation",
    "Mutual Information",
    "Chi-Square",
    "ANOVA F-Test",
    "SelectKBest",
    "RFE",
    "SFS",
    "Lasso",
    "Random Forest",
    "XGBoost",
]

app = FastAPI(title="50 Startups Interactive ML Dashboard", version="2.0.0")
model = joblib.load(MODEL_FILE) if MODEL_FILE.exists() else None


class StartupInput(BaseModel):
    rd_spend: float = Field(ge=0, examples=[100000])
    administration: float = Field(ge=0, examples=[120000])
    marketing_spend: float = Field(ge=0, examples=[250000])
    state: str = Field(examples=["California"])


class DashboardRequest(StartupInput):
    model_name: Literal["Linear Regression", "Random Forest", "XGBoost"] = (
        "Linear Regression"
    )
    algorithm: str = "Correlation"
    feature_count: int = Field(default=1, ge=1, le=5)
    test_size: float = Field(default=0.2, ge=0.1, le=0.4)
    random_state: int = Field(default=42, ge=0, le=10000)
    n_estimators: int = Field(default=300, ge=50, le=1000)
    max_depth: int = Field(default=3, ge=1, le=12)
    learning_rate: float = Field(default=0.05, ge=0.01, le=0.5)


def clean_feature_name(feature: str) -> str:
    return feature.replace("numeric__", "").replace("state__", "")


def create_regressor(request: DashboardRequest):
    if request.model_name == "Random Forest":
        return RandomForestRegressor(
            n_estimators=request.n_estimators,
            max_depth=request.max_depth,
            random_state=request.random_state,
        )
    if request.model_name == "XGBoost":
        return XGBRegressor(
            n_estimators=request.n_estimators,
            max_depth=request.max_depth,
            learning_rate=request.learning_rate,
            random_state=request.random_state,
            n_jobs=1,
        )
    return LinearRegression()


@app.get("/", response_class=FileResponse)
def dashboard() -> FileResponse:
    if not DASHBOARD_FILE.exists():
        raise HTTPException(status_code=404, detail="dashboard.html not found")
    return FileResponse(DASHBOARD_FILE)


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok" if DATA_FILE.exists() else "dataset_not_found",
        "saved_model": model is not None,
    }


@app.get("/api/options")
def options() -> dict[str, object]:
    return {
        "algorithms": ALGORITHMS,
        "models": ["Linear Regression", "Random Forest", "XGBoost"],
        "states": sorted(STATES),
    }


@app.post("/api/analyze")
def analyze(request: DashboardRequest) -> dict[str, object]:
    if request.algorithm not in ALGORITHMS:
        raise HTTPException(status_code=422, detail="Unknown feature algorithm")
    if request.state not in STATES:
        raise HTTPException(status_code=422, detail="Unknown state")
    if not DATA_FILE.exists():
        raise HTTPException(status_code=503, detail="50_Startups.csv not found")

    data = pd.read_csv(DATA_FILE)
    features, target = data[RAW_FEATURES], data[TARGET]
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=request.test_size,
        random_state=request.random_state,
    )

    feature_selection = analyze_feature_selection(x_train, y_train)
    rankings = build_feature_rankings(feature_selection)
    selected = rankings[request.algorithm][: request.feature_count]

    preprocessor = build_pipeline().named_steps["preprocessor"]
    transformed_train = preprocessor.fit_transform(x_train)
    transformed_test = preprocessor.transform(x_test)
    feature_names = preprocessor.get_feature_names_out()
    train_frame = pd.DataFrame(transformed_train, columns=feature_names)
    test_frame = pd.DataFrame(transformed_test, columns=feature_names)

    regressor = create_regressor(request)
    regressor.fit(train_frame[selected], y_train)
    predictions = regressor.predict(test_frame[selected])

    input_row = pd.DataFrame(
        [
            {
                "R&D Spend": request.rd_spend,
                "Administration": request.administration,
                "Marketing Spend": request.marketing_spend,
                "State": request.state,
            }
        ]
    )
    transformed_input = pd.DataFrame(
        preprocessor.transform(input_row), columns=feature_names
    )
    predicted_profit = float(regressor.predict(transformed_input[selected])[0])

    ranking_rows = [
        {
            "rank": rank,
            **{
                algorithm: clean_feature_name(ranked_features[rank - 1])
                for algorithm, ranked_features in rankings.items()
            },
        }
        for rank in range(1, len(feature_names) + 1)
    ]

    return {
        "metrics": {
            "rmse": float(mean_squared_error(y_test, predictions) ** 0.5),
            "r_squared": float(r2_score(y_test, predictions)),
            "training_rows": len(x_train),
            "test_rows": len(x_test),
        },
        "selection": {
            "algorithm": request.algorithm,
            "model": request.model_name,
            "selected_features": [clean_feature_name(name) for name in selected],
        },
        "predicted_profit": predicted_profit,
        "prediction_points": [
            {"actual": float(actual), "predicted": float(predicted)}
            for actual, predicted in zip(y_test, predictions)
        ],
        "rankings": ranking_rows,
        "algorithms": ALGORITHMS,
    }


@app.post("/predict")
def predict(startup: StartupInput) -> dict[str, object]:
    if model is None:
        raise HTTPException(status_code=503, detail="Run the training script first.")
    if startup.state not in STATES:
        raise HTTPException(
            status_code=422, detail=f"State must be one of {sorted(STATES)}"
        )

    row = pd.DataFrame(
        [
            {
                "R&D Spend": startup.rd_spend,
                "Administration": startup.administration,
                "Marketing Spend": startup.marketing_spend,
                "State": startup.state,
            }
        ]
    )
    profit = float(model.predict(row)[0])
    return {"predicted_profit": round(profit, 2), "input": startup.model_dump()}

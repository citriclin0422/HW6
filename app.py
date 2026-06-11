"""FastAPI deployment for the trained 50 Startups profit model."""

from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


MODEL_FILE = Path(__file__).with_name("artifacts") / "best_model.joblib"
STATES = {"California", "Florida", "New York"}

app = FastAPI(title="50 Startups Profit Predictor", version="1.0.0")
model = joblib.load(MODEL_FILE) if MODEL_FILE.exists() else None


class StartupInput(BaseModel):
    rd_spend: float = Field(ge=0, examples=[100000])
    administration: float = Field(ge=0, examples=[120000])
    marketing_spend: float = Field(ge=0, examples=[250000])
    state: str = Field(examples=["California"])


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "Use POST /predict or open /docs"}


@app.get("/health")
def health() -> dict[str, object]:
    return {"status": "ok" if model is not None else "model_not_found"}


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

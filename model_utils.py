"""Reusable transformations shared by training and deployment."""

import pandas as pd


def add_engineered_features(features: pd.DataFrame) -> pd.DataFrame:
    """Add spending features while preserving raw model inputs."""
    result = features.copy()
    spending_columns = ["R&D Spend", "Administration", "Marketing Spend"]
    result["Total Spend"] = result[spending_columns].sum(axis=1)
    result["R&D Ratio"] = result["R&D Spend"] / result["Total Spend"].replace(0, 1)
    return result

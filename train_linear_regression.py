"""CRISP-DM pipeline for predicting profit in the 50 Startups dataset."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.ticker import MultipleLocator
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE, SelectKBest, f_regression
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parent
DATA_FILES = (BASE_DIR / "data.csv", BASE_DIR / "50_Startups.csv")
ARTIFACT_DIR = BASE_DIR / "artifacts"
MODEL_FILE = BASE_DIR / "startup_profit_model.pkl"
API_MODEL_FILE = ARTIFACT_DIR / "best_model.joblib"
TARGET = "Profit"
NUMERIC_FEATURES = ["R&D Spend", "Administration", "Marketing Spend"]
CATEGORICAL_FEATURES = ["State"]
RAW_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
RANDOM_STATE = 42
FEATURE_COMPARISON_ORDER = [
    "R&D Spend",
    "Marketing Spend",
    "State_New York",
    "State_Florida",
    "State_California",
]


def load_and_understand_data() -> pd.DataFrame:
    """CRISP-DM 1-2: load, validate, and summarize the dataset."""
    data_file = next((path for path in DATA_FILES if path.exists()), None)
    if data_file is None:
        names = ", ".join(path.name for path in DATA_FILES)
        raise FileNotFoundError(f"Dataset not found. Expected one of: {names}")

    data = pd.read_csv(data_file)
    expected_columns = RAW_FEATURES + [TARGET]
    if set(data.columns) != set(expected_columns):
        raise ValueError(
            f"Expected columns {expected_columns}, found {data.columns.tolist()}"
        )

    print("=== CRISP-DM 1: Business Understanding ===")
    print("Objective: predict startup Profit from spending and State.\n")
    print("=== CRISP-DM 2: Data Understanding / EDA ===")
    print(f"Dataset: {data_file.name}")
    print(f"Shape: {data.shape}")
    print("\nData types:")
    print(data.dtypes.to_string())
    print(f"\nMissing values:\n{data.isna().sum().to_string()}")
    print(f"\nDuplicate rows: {data.duplicated().sum()}")
    print(f"\nDescriptive statistics:\n{data.describe().round(2).to_string()}")
    print(
        "\nCorrelation with Profit:\n"
        + data.select_dtypes("number")
        .corr()[TARGET]
        .sort_values(ascending=False)
        .round(4)
        .to_string()
    )
    return data


def create_eda_charts(data: pd.DataFrame) -> None:
    """Save the EDA visualizations described in the project log."""
    ARTIFACT_DIR.mkdir(exist_ok=True)
    numeric = data.select_dtypes("number")

    axes = numeric.hist(figsize=(11, 8), bins=10)
    figure = axes[0, 0].figure
    figure.suptitle("50 Startups: Numeric Distributions")
    figure.tight_layout()
    figure.savefig(ARTIFACT_DIR / "eda_histograms.png", dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(10, 5))
    numeric.boxplot(ax=axis, rot=20)
    axis.set_title("Numeric Feature Boxplots")
    figure.tight_layout()
    figure.savefig(ARTIFACT_DIR / "eda_boxplots.png", dpi=150)
    plt.close(figure)

    correlation = numeric.corr()
    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1)
    axis.set_xticks(range(len(correlation)), correlation.columns, rotation=35, ha="right")
    axis.set_yticks(range(len(correlation)), correlation.columns)
    for row in range(len(correlation)):
        for column in range(len(correlation)):
            axis.text(
                column,
                row,
                f"{correlation.iloc[row, column]:.2f}",
                ha="center",
                va="center",
            )
    axis.set_title("Correlation Matrix")
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(ARTIFACT_DIR / "eda_correlation_heatmap.png", dpi=150)
    plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    for axis, feature in zip(axes, NUMERIC_FEATURES):
        axis.scatter(data[feature], data[TARGET], alpha=0.8)
        axis.set(title=f"Profit vs {feature}", xlabel=feature, ylabel=TARGET)
    figure.tight_layout()
    figure.savefig(ARTIFACT_DIR / "eda_profit_scatterplots.png", dpi=150)
    plt.close(figure)

    state_order = sorted(data["State"].unique())
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    data["State"].value_counts().reindex(state_order).plot.bar(ax=axes[0])
    axes[0].set(title="Startup Count by State", xlabel="State", ylabel="Count")
    axes[1].boxplot(
        [data.loc[data["State"] == state, TARGET] for state in state_order],
        tick_labels=state_order,
    )
    axes[1].set(title="Profit by State", xlabel="State", ylabel=TARGET)
    figure.tight_layout()
    figure.savefig(ARTIFACT_DIR / "eda_state_analysis.png", dpi=150)
    plt.close(figure)

    correlation.to_csv(ARTIFACT_DIR / "eda_correlation_matrix.csv")


def create_workflow_diagram() -> None:
    """Render an overview of the implemented CRISP-DM workflow."""
    ARTIFACT_DIR.mkdir(exist_ok=True)
    figure, axis = plt.subplots(figsize=(16, 10))
    axis.set_xlim(0, 16)
    axis.set_ylim(0, 10)
    axis.axis("off")

    def add_box(
        x: float,
        y: float,
        width: float,
        height: float,
        title: str,
        details: str,
        color: str,
    ) -> None:
        box = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.04,rounding_size=0.12",
            linewidth=1.5,
            edgecolor="#263238",
            facecolor=color,
        )
        axis.add_patch(box)
        axis.text(
            x + width / 2,
            y + height * 0.68,
            title,
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
        )
        axis.text(
            x + width / 2,
            y + height * 0.32,
            details,
            ha="center",
            va="center",
            fontsize=8.5,
            linespacing=1.3,
        )

    def add_arrow(start: tuple[float, float], end: tuple[float, float]) -> None:
        axis.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=16,
                linewidth=1.5,
                color="#455A64",
                connectionstyle="arc3,rad=0",
            )
        )

    axis.text(
        8,
        9.65,
        "50 Startups Profit Prediction - CRISP-DM Workflow",
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
    )

    add_box(0.5, 7.5, 2.4, 1.35, "1. Business Understanding",
            "Predict Profit\nIdentify useful features", "#E3F2FD")
    add_box(3.35, 7.5, 2.4, 1.35, "2. Data Understanding",
            "50_Startups.csv\nEDA and data-quality checks", "#E8F5E9")
    add_box(6.2, 7.5, 2.4, 1.35, "3. Data Preparation",
            "80/20 split\nScaling and one-hot encoding", "#FFF8E1")
    add_box(9.05, 7.5, 2.4, 1.35, "4. Feature Selection",
            "Rank and evaluate\ntop 1-5 features", "#F3E5F5")
    add_box(11.9, 7.5, 2.4, 1.35, "5. Modeling",
            "Linear Regression\nPipeline training", "#FCE4EC")

    for start_x in (2.9, 5.75, 8.6, 11.45):
        add_arrow((start_x, 8.18), (start_x + 0.45, 8.18))

    selectors = [
        ("Correlation", 0.65),
        ("SelectKBest\nF-Regression", 3.0),
        ("Recursive Feature\nElimination", 5.35),
        ("Lasso\nRegression", 7.7),
        ("Random Forest\nImportance", 10.05),
    ]
    for label, x in selectors:
        add_box(x, 5.25, 1.9, 1.15, label, "Feature ranking", "#F3E5F5")
        add_arrow((10.25, 7.5), (x + 0.95, 6.4))

    add_box(12.4, 5.25, 2.9, 1.15, "Performance Comparison",
            "Feature counts 1-5\nRMSE and R-squared", "#E0F7FA")
    for _, x in selectors:
        add_arrow((x + 1.9, 5.82), (12.4, 5.82))
    add_arrow((13.1, 7.5), (13.85, 6.4))

    add_box(1.0, 2.7, 3.2, 1.35, "7. Deployment",
            "Saved fitted pipeline\nFastAPI POST /predict", "#E8EAF6")
    add_box(6.4, 2.7, 3.2, 1.35, "Artifacts and Visualizations",
            "CSV results, EDA charts\nAll-in-one and feature-count figures", "#FFF3E0")
    add_box(11.8, 2.7, 3.2, 1.35, "6. Evaluation",
            "Holdout test metrics\nShuffled 5-fold cross-validation", "#E0F2F1")
    add_arrow((13.35, 5.25), (13.4, 4.05))
    add_arrow((11.8, 3.38), (9.6, 3.38))
    add_arrow((6.4, 3.38), (4.2, 3.38))

    add_box(4.4, 0.65, 7.2, 1.15, "Final Insight",
            "R&D Spend is the strongest predictor; Marketing Spend adds useful "
            "cross-validation performance.", "#ECEFF1")
    add_arrow((2.6, 2.7), (5.8, 1.8))
    add_arrow((8.0, 2.7), (8.0, 1.8))
    add_arrow((13.4, 2.7), (10.2, 1.8))

    figure.tight_layout()
    figure.savefig(ARTIFACT_DIR / "workflow.png", dpi=150, bbox_inches="tight")
    plt.close(figure)


def build_pipeline() -> Pipeline:
    """Build a leakage-safe preprocessing and linear-regression pipeline."""
    preprocessor = ColumnTransformer(
        [
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            (
                "state",
                OneHotEncoder(
                    drop="first", handle_unknown="ignore", sparse_output=False
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("regressor", LinearRegression()),
        ]
    )


def analyze_feature_selection(
    x_train: pd.DataFrame, y_train: pd.Series
) -> pd.DataFrame:
    """Run the five feature-selection methods specified in the project log."""
    preprocessor = build_pipeline().named_steps["preprocessor"]
    transformed = preprocessor.fit_transform(x_train)
    feature_names = preprocessor.get_feature_names_out()
    transformed_frame = pd.DataFrame(
        transformed, columns=feature_names, index=x_train.index
    )

    pearson = transformed_frame.apply(lambda column: column.corr(y_train))
    select_k_best = SelectKBest(score_func=f_regression, k="all").fit(
        transformed_frame, y_train
    )
    rfe = RFE(LinearRegression(), n_features_to_select=1).fit(
        transformed_frame, y_train
    )
    lasso = Lasso(alpha=100.0, max_iter=100000).fit(transformed_frame, y_train)
    forest = RandomForestRegressor(
        n_estimators=500, random_state=RANDOM_STATE
    ).fit(transformed_frame, y_train)

    return (
        pd.DataFrame(
            {
                "Feature": feature_names,
                "Pearson Correlation": pearson.values,
                "F-Regression Score": select_k_best.scores_,
                "RFE Rank": rfe.ranking_,
                "Lasso Coefficient": lasso.coef_,
                "Random Forest Importance": forest.feature_importances_,
            }
        )
        .sort_values("RFE Rank")
        .reset_index(drop=True)
    )


def create_feature_selection_performance_figure(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    feature_selection: pd.DataFrame,
) -> pd.DataFrame:
    """Evaluate the top-k features chosen by each selection algorithm."""
    preprocessor = build_pipeline().named_steps["preprocessor"]
    transformed_train = preprocessor.fit_transform(x_train)
    transformed_test = preprocessor.transform(x_test)
    feature_names = preprocessor.get_feature_names_out()
    train_frame = pd.DataFrame(
        transformed_train, columns=feature_names, index=x_train.index
    )
    test_frame = pd.DataFrame(
        transformed_test, columns=feature_names, index=x_test.index
    )

    rankings = {
        "Correlation Analysis": feature_selection.assign(
            score=feature_selection["Pearson Correlation"].abs()
        ).sort_values("score", ascending=False)["Feature"].tolist(),
        "SelectKBest F-Regression": feature_selection.sort_values(
            "F-Regression Score", ascending=False
        )["Feature"].tolist(),
        "Recursive Feature Elimination": feature_selection.sort_values(
            "RFE Rank"
        )["Feature"].tolist(),
        "Lasso Regression": feature_selection.assign(
            score=feature_selection["Lasso Coefficient"].abs()
        ).sort_values("score", ascending=False)["Feature"].tolist(),
        "Random Forest Importance": feature_selection.sort_values(
            "Random Forest Importance", ascending=False
        )["Feature"].tolist(),
    }

    rows = []
    for algorithm, ranked_features in rankings.items():
        for number_of_features in range(1, len(ranked_features) + 1):
            selected = ranked_features[:number_of_features]
            evaluator = LinearRegression().fit(train_frame[selected], y_train)
            predictions = evaluator.predict(test_frame[selected])
            rows.append(
                {
                    "Algorithm": algorithm,
                    "Number of Features": number_of_features,
                    "Selected Features": selected,
                    "RMSE": mean_squared_error(y_test, predictions) ** 0.5,
                    "R-squared": r2_score(y_test, predictions),
                }
            )

    results = pd.DataFrame(rows)
    ARTIFACT_DIR.mkdir(exist_ok=True)
    results.assign(
        **{"Selected Features": results["Selected Features"].map(str)}
    ).to_csv(ARTIFACT_DIR / "feature_selection_performance_allinone.csv", index=False)

    figure = plt.figure(figsize=(14, 16))
    grid = figure.add_gridspec(2, 2, height_ratios=[3.2, 3.6])
    axes = [figure.add_subplot(grid[0, 0]), figure.add_subplot(grid[0, 1])]
    table_axis = figure.add_subplot(grid[1, :])
    for algorithm, algorithm_results in results.groupby("Algorithm", sort=False):
        axes[0].plot(
            algorithm_results["Number of Features"],
            algorithm_results["RMSE"],
            marker="o",
            label=algorithm,
        )
        axes[1].plot(
            algorithm_results["Number of Features"],
            algorithm_results["R-squared"],
            marker="o",
            label=algorithm,
        )

    axes[0].set(
        title="Feature Selection Performance: RMSE",
        xlabel="Number of Selected Features",
        ylabel="Test RMSE",
    )
    axes[1].set(
        title="Feature Selection Performance: R-squared",
        xlabel="Number of Selected Features",
        ylabel="Test R-squared",
    )
    for axis in axes:
        axis.set_xticks(range(1, len(feature_names) + 1))
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)

    reference_rows = results.reset_index(drop=True)

    def display_features(features: list[str]) -> str:
        return ", ".join(
            feature.replace("numeric__", "").replace("state__", "")
            for feature in features
        )

    table_axis.axis("off")
    table_axis.set_title(
        "Reference Results for Feature Counts 1-5", fontsize=11, pad=8
    )
    summary_table = table_axis.table(
        cellText=[
            [
                row["Algorithm"],
                row["Number of Features"],
                display_features(row["Selected Features"]),
                f"{row['RMSE']:,.2f}",
                f"{row['R-squared']:.4f}",
            ]
            for _, row in reference_rows.iterrows()
        ],
        colLabels=[
            "Algorithm",
            "Feature Count",
            "Selected Features",
            "RMSE",
            "R-squared",
        ],
        cellLoc="left",
        colLoc="left",
        loc="center",
        colWidths=[0.22, 0.13, 0.40, 0.12, 0.13],
    )
    summary_table.auto_set_font_size(False)
    summary_table.set_fontsize(6)
    summary_table.scale(1, 1.05)
    for column in range(5):
        summary_table[(0, column)].set_text_props(weight="bold")

    figure.suptitle("Top 5 Feature Selection Algorithms - Performance Comparison")
    figure.tight_layout()
    figure.savefig(
        ARTIFACT_DIR / "feature_selection_performance_allinone.png", dpi=150
    )
    plt.close(figure)
    return results


def create_feature_count_comparison(data: pd.DataFrame) -> pd.DataFrame:
    """Compare cumulative expert-ranked features and render the reference diagram."""
    encoded = pd.get_dummies(
        data[RAW_FEATURES], columns=CATEGORICAL_FEATURES, dtype=float
    )
    cross_validation = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows = []

    for number_of_features in range(1, len(FEATURE_COMPARISON_ORDER) + 1):
        selected = FEATURE_COMPARISON_ORDER[:number_of_features]
        scores = cross_validate(
            LinearRegression(),
            encoded[selected],
            data[TARGET],
            cv=cross_validation,
            scoring={
                "rmse": "neg_root_mean_squared_error",
                "r_squared": "r2",
            },
        )
        rows.append(
            {
                "Number of Features": number_of_features,
                "Selected Features": selected,
                "RMSE": -scores["test_rmse"].mean(),
                "R-squared": scores["test_r_squared"].mean(),
            }
        )

    results = pd.DataFrame(rows)
    ARTIFACT_DIR.mkdir(exist_ok=True)
    results.assign(
        **{"Selected Features": results["Selected Features"].map(str)}
    ).to_csv(ARTIFACT_DIR / "feature_count_comparison.csv", index=False)

    figure = plt.figure(figsize=(8, 6.5))
    grid = figure.add_gridspec(2, 2, height_ratios=[3.2, 1.25])
    rmse_axis = figure.add_subplot(grid[0, 0])
    r_squared_axis = figure.add_subplot(grid[0, 1])
    table_axis = figure.add_subplot(grid[1, :])

    feature_counts = results["Number of Features"]
    rmse_axis.plot(feature_counts, results["RMSE"], marker="o", markersize=4)
    rmse_axis.set(
        title="RMSE by Number of Features",
        xlabel="Number of Features",
        ylabel="RMSE",
    )
    rmse_axis.set_ylim(8200, 9400)
    rmse_axis.yaxis.set_major_locator(MultipleLocator(200))
    r_squared_axis.plot(
        feature_counts, results["R-squared"], marker="o", markersize=4
    )
    r_squared_axis.set(
        title="R-squared by Number of Features",
        xlabel="Number of Features",
        ylabel="R-squared",
    )
    r_squared_axis.set_ylim(0.932, 0.940)
    r_squared_axis.yaxis.set_major_locator(MultipleLocator(0.002))

    table_axis.axis("off")
    table_rows = [
        [
            row["Number of Features"],
            str(row["Selected Features"]),
            f"{row['RMSE']:.6f}",
            f"{row['R-squared']:.6f}",
        ]
        for _, row in results.iterrows()
    ]
    table = table_axis.table(
        cellText=table_rows,
        colLabels=results.columns,
        cellLoc="left",
        colLoc="left",
        loc="center",
        colWidths=[0.18, 0.62, 0.10, 0.10],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1, 1.35)
    for column in range(len(results.columns)):
        table[(0, column)].set_text_props(weight="bold")

    figure.tight_layout()
    table_axis.set_position([0.02, 0.02, 0.96, 0.22])
    png_file = ARTIFACT_DIR / "feature_count_comparison.png"
    gif_file = ARTIFACT_DIR / "final_output_diagram.gif"
    figure.savefig(png_file, dpi=100)
    plt.close(figure)
    with Image.open(png_file) as image:
        image.save(gif_file, format="GIF")

    return results


def main() -> None:
    data = load_and_understand_data()
    create_eda_charts(data)
    create_workflow_diagram()

    print("\n=== CRISP-DM 3: Data Preparation ===")
    print("State: one-hot encoded with the first category dropped.")
    print("Numeric spending features: standardized.")
    print("Split: 80% training / 20% testing with random_state=42.")

    features, target = data[RAW_FEATURES], data[TARGET]
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=RANDOM_STATE
    )

    feature_selection = analyze_feature_selection(x_train, y_train)
    print("\n=== CRISP-DM 3.11: Feature Selection Analysis ===")
    print(feature_selection.round(4).to_string(index=False))

    selection_performance = create_feature_selection_performance_figure(
        x_train, x_test, y_train, y_test, feature_selection
    )
    best_selection = selection_performance.loc[selection_performance["RMSE"].idxmin()]
    print("\n=== Feature Selection Algorithm Performance ===")
    print(
        f"Best: {best_selection['Algorithm']} with "
        f"{best_selection['Number of Features']} features, "
        f"RMSE ${best_selection['RMSE']:,.2f}, "
        f"R-squared {best_selection['R-squared']:.4f}"
    )

    feature_count_comparison = create_feature_count_comparison(data)
    print("\n=== Feature Count Comparison ===")
    print(feature_count_comparison.round(6).to_string(index=False))

    print("\n=== CRISP-DM 4 & 5: Modeling and Evaluation ===")
    model = build_pipeline()
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    rmse = float(mean_squared_error(y_test, predictions) ** 0.5)
    r2 = float(r2_score(y_test, predictions))
    print(f"RMSE: ${rmse:,.2f}")
    print(f"R-Squared (R2): {r2:.4f}")

    ARTIFACT_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_FILE)
    joblib.dump(model, API_MODEL_FILE)
    feature_selection.to_csv(
        ARTIFACT_DIR / "feature_selection_results.csv", index=False
    )
    (ARTIFACT_DIR / "model_metadata.json").write_text(
        json.dumps(
            {
                "model": "Multiple Linear Regression",
                "test_rmse": rmse,
                "test_r2": r2,
                "random_state": RANDOM_STATE,
                "training_rows": len(x_train),
                "test_rows": len(x_test),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n=== CRISP-DM 6: Deployment ===")
    print(f"Saved complete fitted pipeline: {MODEL_FILE}")
    print(f"Saved API-compatible copy: {API_MODEL_FILE}")


if __name__ == "__main__":
    main()

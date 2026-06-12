"""CRISP-DM pipeline for predicting profit in the 50 Startups dataset.

繁體中文說明：
此程式依照 CRISP-DM（跨產業資料探勘標準流程）建立新創公司獲利預測模型。
完整流程包含資料載入與驗證、探索式資料分析（EDA）、資料前處理、五種特徵選擇
方法、線性迴歸模型訓練與評估，以及儲存可供後續 API 使用的模型與分析圖表。
"""

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
from sklearn.feature_selection import RFE, SelectKBest, chi2, f_regression, mutual_info_regression
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from xgboost import XGBRegressor


# -------------------------------------------------
# CRISP-DM Step 1: Business Understanding
# -------------------------------------------------
# 商業理解：本專案的目標是根據新創公司的研發、行政、行銷支出與所在州別，
# 建立可預測公司獲利（Profit）的迴歸模型，並找出最具預測能力的特徵。

# 專案路徑與輸出位置：所有相對路徑皆以此 Python 檔案所在資料夾為基準，
# 因此無論從哪一個工作目錄執行程式，都能正確找到資料並輸出結果。
BASE_DIR = Path(__file__).resolve().parent
DATA_FILES = (BASE_DIR / "data.csv", BASE_DIR / "50_Startups.csv")
ARTIFACT_DIR = BASE_DIR / "artifacts"
MODEL_FILE = BASE_DIR / "startup_profit_model.pkl"
API_MODEL_FILE = ARTIFACT_DIR / "best_model.joblib"

# 資料欄位設定：Profit 是預測目標；支出欄位是數值特徵；State 是類別特徵。
# RAW_FEATURES 代表模型可直接從原始資料取得的全部輸入欄位。
TARGET = "Profit"
NUMERIC_FEATURES = ["R&D Spend", "Administration", "Marketing Spend"]
CATEGORICAL_FEATURES = ["State"]
RAW_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# 固定隨機種子可讓資料切分、交叉驗證與隨機森林的重要性結果具有可重現性。
RANDOM_STATE = 42

# 專家指定的累加特徵順序，用來觀察依序加入特徵後，模型效能如何變化。
FEATURE_COMPARISON_ORDER = [
    "R&D Spend",
    "Marketing Spend",
    "State_New York",
    "State_Florida",
    "State_California",
]


# -------------------------------------------------
# CRISP-DM Step 2: Data Understanding
# -------------------------------------------------
# 資料理解：載入並驗證資料，檢查資料品質、統計特性、欄位相關性，
# 並建立探索式資料分析圖表與完整 CRISP-DM 流程圖。

def load_and_understand_data() -> pd.DataFrame:
    """CRISP-DM 1-2: load, validate, and summarize the dataset.

    繁體中文說明：
    執行 CRISP-DM 的「商業理解」與「資料理解」階段。函式會先尋找可用的 CSV
    資料檔，確認欄位是否完整且沒有多餘欄位，再輸出資料筆數、資料型態、缺失值、
    重複資料、描述統計，以及各數值欄位與 Profit 的相關係數，協助初步判斷資料
    品質與可能的重要預測因子。
    """
    # 依照 DATA_FILES 中的優先順序，取得第一個實際存在的資料檔。
    data_file = next((path for path in DATA_FILES if path.exists()), None)
    if data_file is None:
        names = ", ".join(path.name for path in DATA_FILES)
        raise FileNotFoundError(f"Dataset not found. Expected one of: {names}")

    # 載入資料後，使用集合比對確認欄位名稱完全符合模型預期。
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
    """Save the EDA visualizations described in the project log.

    繁體中文說明：
    建立並儲存探索式資料分析（EDA）圖表，包括數值欄位直方圖、箱型圖、相關係數
    熱圖、各支出欄位與獲利的散佈圖，以及不同州別的公司數量與獲利分布。這些圖表
    可用來觀察資料分布、離群值、變數關係與類別差異；相關係數矩陣也會另存成 CSV。
    """
    ARTIFACT_DIR.mkdir(exist_ok=True)
    # 僅選取數值欄位，避免類別型的 State 無法直接進行數值統計與相關分析。
    numeric = data.select_dtypes("number")

    # 直方圖：查看各數值欄位的分布形狀、集中區域與偏態。
    axes = numeric.hist(figsize=(11, 8), bins=10)
    figure = axes[0, 0].figure
    figure.suptitle("50 Startups: Numeric Distributions")
    figure.tight_layout()
    figure.savefig(ARTIFACT_DIR / "eda_histograms.png", dpi=150)
    plt.close(figure)

    # 箱型圖：快速辨識中位數、四分位距，以及可能存在的離群值。
    figure, axis = plt.subplots(figsize=(10, 5))
    numeric.boxplot(ax=axis, rot=20)
    axis.set_title("Numeric Feature Boxplots")
    figure.tight_layout()
    figure.savefig(ARTIFACT_DIR / "eda_boxplots.png", dpi=150)
    plt.close(figure)

    # 相關係數熱圖：呈現數值欄位兩兩之間線性關係的方向與強度。
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

    # 散佈圖：分別檢視各項支出與 Profit 之間是否具有線性趨勢。
    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    for axis, feature in zip(axes, NUMERIC_FEATURES):
        axis.scatter(data[feature], data[TARGET], alpha=0.8)
        axis.set(title=f"Profit vs {feature}", xlabel=feature, ylabel=TARGET)
    figure.tight_layout()
    figure.savefig(ARTIFACT_DIR / "eda_profit_scatterplots.png", dpi=150)
    plt.close(figure)

    # 州別分析：左圖比較各州樣本數，右圖比較各州 Profit 的分布。
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
    """Render an overview of the implemented CRISP-DM workflow.

    繁體中文說明：
    繪製本專案完整 CRISP-DM 工作流程圖。圖中呈現從商業理解、資料理解、資料準備、
    特徵選擇、建模、評估到部署的執行順序，也列出五種特徵排序方法與最終產出，
    方便讀者快速掌握此專案的分析架構。
    """
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
        """Add one rounded information box to the workflow diagram.

        繁體中文說明：在流程圖指定座標加入圓角資訊方塊，並設定標題、詳細文字與底色。
        """
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
        """Connect two workflow elements with a directional arrow.

        繁體中文說明：使用帶箭頭的線段連接兩個流程元素，以表示執行方向或資料流向。
        """
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


# -------------------------------------------------
# CRISP-DM Step 3: Data Preparation
# -------------------------------------------------
# 資料準備：將數值欄位標準化、將類別欄位進行 One-Hot Encoding，
# 並以 Pipeline 封裝前處理與模型，避免訓練資料與測試資料之間發生資料洩漏。

def build_pipeline() -> Pipeline:
    """Build a leakage-safe preprocessing and linear-regression pipeline.

    繁體中文說明：
    建立不易發生資料洩漏（data leakage）的完整 sklearn Pipeline。數值欄位會以
    StandardScaler 標準化；State 則以 OneHotEncoder 轉換成虛擬變數。前處理器與
    LinearRegression 被包在同一條 Pipeline 中，因此模型訓練或交叉驗證時，前處理
    參數只會由訓練資料估計，不會提前看到測試資料。
    """
    preprocessor = ColumnTransformer(
        [
            # 將數值特徵轉換為平均數約為 0、標準差約為 1 的尺度。
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            (
                "state",
                # One-hot encoding 將 State 類別轉成數值欄位。
                # drop="first" 移除第一類以避免完全共線性；未知州別則忽略而不報錯。
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


# -------------------------------------------------
# CRISP-DM Step 4: Model Selection
# -------------------------------------------------
# 模型選擇：透過五種特徵選擇方法評估各輸入欄位的重要程度，
# 找出適合提供給線性迴歸模型的特徵組合。

def _analyze_feature_selection_legacy(
    x_train: pd.DataFrame, y_train: pd.Series
) -> pd.DataFrame:
    """Run the five feature-selection methods specified in the project log.

    繁體中文說明：
    僅使用訓練資料執行五種特徵選擇／排序方法：Pearson 相關係數、SelectKBest
    F-Regression、遞迴特徵消除（RFE）、Lasso 迴歸，以及隨機森林特徵重要性。
    各方法衡量重要性的方式不同，因此將結果整理成同一個 DataFrame，便於比較。
    """
    # 先對訓練資料擬合前處理器，再把轉換結果還原成有欄名的 DataFrame。
    preprocessor = build_pipeline().named_steps["preprocessor"]
    transformed = preprocessor.fit_transform(x_train)
    feature_names = preprocessor.get_feature_names_out()
    transformed_frame = pd.DataFrame(
        transformed, columns=feature_names, index=x_train.index
    )

    # Pearson：衡量單一特徵與目標值之間的線性相關程度。
    pearson = transformed_frame.apply(lambda column: column.corr(y_train))
    # F-Regression：使用單變量線性迴歸的 F 統計量評估每個特徵。
    select_k_best = SelectKBest(score_func=f_regression, k="all").fit(
        transformed_frame, y_train
    )
    # RFE：反覆移除最不重要的特徵；排名 1 代表最後保留下來的特徵。
    rfe = RFE(LinearRegression(), n_features_to_select=1).fit(
        transformed_frame, y_train
    )
    # Lasso：透過 L1 正則化將較不重要特徵的係數壓縮至接近或等於 0。
    lasso = Lasso(alpha=100.0, max_iter=100000).fit(transformed_frame, y_train)
    # 隨機森林：依據樹模型分裂過程帶來的誤差改善計算特徵重要性。
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


def sequential_forward_ranking(
    features: pd.DataFrame, target: pd.Series
) -> list[str]:
    """Rank all features using sequential forward selection and CV RMSE."""
    remaining = features.columns.tolist()
    selected: list[str] = []
    cross_validation = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    while remaining:
        candidate_scores = {}
        for candidate in remaining:
            candidate_set = selected + [candidate]
            scores = cross_validate(
                LinearRegression(),
                features[candidate_set],
                target,
                cv=cross_validation,
                scoring="neg_root_mean_squared_error",
            )
            candidate_scores[candidate] = -scores["test_score"].mean()

        best_candidate = min(candidate_scores, key=candidate_scores.get)
        selected.append(best_candidate)
        remaining.remove(best_candidate)

    return selected


def analyze_feature_selection(
    x_train: pd.DataFrame, y_train: pd.Series
) -> pd.DataFrame:
    """Run ten feature-selection methods and return their scores and ranks."""
    preprocessor = build_pipeline().named_steps["preprocessor"]
    transformed = preprocessor.fit_transform(x_train)
    feature_names = preprocessor.get_feature_names_out()
    transformed_frame = pd.DataFrame(
        transformed, columns=feature_names, index=x_train.index
    )

    pearson = transformed_frame.apply(lambda column: column.corr(y_train))
    mutual_information = mutual_info_regression(
        transformed_frame, y_train, random_state=RANDOM_STATE
    )

    # Chi-Square requires nonnegative inputs and a categorical target.
    nonnegative_features = MinMaxScaler().fit_transform(transformed_frame)
    profit_classes = pd.qcut(y_train, q=4, labels=False, duplicates="drop")
    chi_square_scores, _ = chi2(nonnegative_features, profit_classes)

    anova_scores, _ = f_regression(transformed_frame, y_train)
    select_k_best = SelectKBest(score_func=f_regression, k="all").fit(
        transformed_frame, y_train
    )
    rfe = RFE(LinearRegression(), n_features_to_select=1).fit(
        transformed_frame, y_train
    )
    sfs_ranking = sequential_forward_ranking(transformed_frame, y_train)
    sfs_ranks = {feature: rank for rank, feature in enumerate(sfs_ranking, start=1)}
    lasso = Lasso(alpha=100.0, max_iter=100000).fit(transformed_frame, y_train)
    forest = RandomForestRegressor(
        n_estimators=500, random_state=RANDOM_STATE
    ).fit(transformed_frame, y_train)
    xgboost = XGBRegressor(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        random_state=RANDOM_STATE,
        n_jobs=1,
    ).fit(transformed_frame, y_train)

    return pd.DataFrame(
        {
            "Feature": feature_names,
            "Pearson Correlation": pearson.values,
            "Mutual Information": mutual_information,
            "Chi-Square Score": chi_square_scores,
            "ANOVA F-Test Score": anova_scores,
            "F-Regression Score": select_k_best.scores_,
            "RFE Rank": rfe.ranking_,
            "SFS Rank": [sfs_ranks[feature] for feature in feature_names],
            "Lasso Coefficient": lasso.coef_,
            "Random Forest Importance": forest.feature_importances_,
            "XGBoost Importance": xgboost.feature_importances_,
        }
    )


def build_feature_rankings(feature_selection: pd.DataFrame) -> dict[str, list[str]]:
    """Convert all ten feature-selection results into best-to-worst rankings."""
    return {
        "Correlation": feature_selection.assign(
            score=feature_selection["Pearson Correlation"].abs()
        ).sort_values("score", ascending=False)["Feature"].tolist(),
        "Mutual Information": feature_selection.sort_values(
            "Mutual Information", ascending=False
        )["Feature"].tolist(),
        "Chi-Square": feature_selection.sort_values(
            "Chi-Square Score", ascending=False
        )["Feature"].tolist(),
        "ANOVA F-Test": feature_selection.sort_values(
            "ANOVA F-Test Score", ascending=False
        )["Feature"].tolist(),
        "SelectKBest": feature_selection.sort_values(
            "F-Regression Score", ascending=False
        )["Feature"].tolist(),
        "RFE": feature_selection.sort_values("RFE Rank")["Feature"].tolist(),
        "SFS": feature_selection.sort_values("SFS Rank")["Feature"].tolist(),
        "Lasso": feature_selection.assign(
            score=feature_selection["Lasso Coefficient"].abs()
        ).sort_values("score", ascending=False)["Feature"].tolist(),
        "Random Forest": feature_selection.sort_values(
            "Random Forest Importance", ascending=False
        )["Feature"].tolist(),
        "XGBoost": feature_selection.sort_values(
            "XGBoost Importance", ascending=False
        )["Feature"].tolist(),
    }


# -------------------------------------------------
# CRISP-DM Step 5: Model Evaluation
# -------------------------------------------------
# 模型評估：比較不同特徵選擇演算法與特徵數量的模型表現，
# 使用 RMSE 與 R-squared 衡量預測誤差及模型解釋能力。

def create_feature_selection_performance_figure(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    feature_selection: pd.DataFrame,
) -> pd.DataFrame:
    """Evaluate the top-k features chosen by each selection algorithm.

    繁體中文說明：
    根據五種特徵選擇方法各自產生的排序，依序取前 1 到全部特徵訓練線性迴歸，
    並使用保留的測試集計算 RMSE 與 R-squared。最後將所有組合的結果輸出成 CSV，
    同時繪製效能折線圖及明細表，協助比較不同方法與特徵數量的優劣。
    """
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

    # 將不同方法的分數統一轉換成「最重要到最不重要」的特徵名稱清單。
    rankings = build_feature_rankings(feature_selection)

    # 對每種排序逐步增加特徵數量，並以相同的線性迴歸模型公平比較效能。
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

    figure = plt.figure(figsize=(24, 14))
    grid = figure.add_gridspec(2, 2, height_ratios=[3.2, 1.8])
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

    def display_features(features: list[str]) -> str:
        """Remove sklearn transformer prefixes for display.

        繁體中文說明：移除 sklearn 自動加入的轉換器前綴，讓圖表中的欄位名稱更易閱讀。
        """
        return ", ".join(
            feature.replace("numeric__", "").replace("state__", "")
            for feature in features
        )

    table_axis.axis("off")
    table_axis.set_title(
        "Feature Rankings by Algorithm",
        fontsize=11,
        pad=8,
    )
    algorithm_labels = [
        f"Algorithm-{index}\n({algorithm})"
        for index, algorithm in enumerate(rankings, start=1)
    ]
    summary_table = table_axis.table(
        cellText=[
            [f"Rank {rank}"]
            + [
                display_features([ranked_features[rank - 1]])
                for ranked_features in rankings.values()
            ]
            for rank in range(1, len(feature_names) + 1)
        ],
        colLabels=["Rank"] + algorithm_labels,
        cellLoc="left",
        colLoc="left",
        loc="center",
        colWidths=[0.07] + [0.093] * len(rankings),
    )
    summary_table.auto_set_font_size(False)
    summary_table.set_fontsize(7.5)
    summary_table.scale(1, 1.55)
    for column in range(len(rankings) + 1):
        summary_table[(0, column)].set_text_props(weight="bold")

    figure.suptitle(
        "Top 10 Feature Selection Algorithms - Test RMSE and R-squared Comparison"
    )
    figure.tight_layout()
    output_files = [
        ARTIFACT_DIR / "feature_selection_performance_allinone.png",
        ARTIFACT_DIR / "feature_selection_10_algorithms_comparison.png",
    ]
    for output_file in output_files:
        figure.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return results


def create_feature_count_comparison(data: pd.DataFrame) -> pd.DataFrame:
    """Compare cumulative expert-ranked features and render the reference diagram.

    繁體中文說明：
    依 FEATURE_COMPARISON_ORDER 指定的專家排序，從第一個特徵開始逐一累加，並使用
    打亂後的五折交叉驗證評估每組特徵的平均 RMSE 與 R-squared。此處另以
    pd.get_dummies 將 State 展開成虛擬變數，最後輸出比較表、PNG 圖與 GIF 圖。
    """
    # 將類別型 State 轉為數值虛擬欄位，讓 LinearRegression 可直接使用。
    encoded = pd.get_dummies(
        data[RAW_FEATURES], columns=CATEGORICAL_FEATURES, dtype=float
    )
    # 五折交叉驗證會輪流使用不同資料折作為驗證集；shuffle 可降低原始排序的影響。
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


# -------------------------------------------------
# CRISP-DM Step 6: Deployment
# -------------------------------------------------
# 部署：執行完整 CRISP-DM 流程，訓練最終模型，並將模型、評估結果、
# 中繼資料與視覺化成果儲存至檔案，供後續 FastAPI 或其他程式載入使用。

def main() -> None:
    """Run the complete training, evaluation, visualization, and export workflow.

    繁體中文說明：
    主程式依序執行資料理解與 EDA、資料切分、特徵選擇比較、模型訓練與測試集評估，
    最後儲存完整 Pipeline、API 使用版本、特徵選擇結果與模型中繼資料。執行此檔案
    時，所有階段的摘要與指標都會輸出至終端機。
    """
    # -------------------------------------------------
    # CRISP-DM Step 1-2: Business and Data Understanding
    # -------------------------------------------------
    # CRISP-DM 1-2：載入、驗證、探索資料，並產生 EDA 與流程圖。
    data = load_and_understand_data()
    create_eda_charts(data)
    create_workflow_diagram()

    # -------------------------------------------------
    # CRISP-DM Step 3: Data Preparation
    # -------------------------------------------------
    print("\n=== CRISP-DM 3: Data Preparation ===")
    print("State: one-hot encoded with the first category dropped.")
    print("Numeric spending features: standardized.")
    print("Split: 80% training / 20% testing with random_state=42.")

    # 將原始資料切分成 80% 訓練集與 20% 測試集；測試集只用於最終效能評估。
    features, target = data[RAW_FEATURES], data[TARGET]
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=RANDOM_STATE
    )

    # -------------------------------------------------
    # CRISP-DM Step 4: Model Selection
    # -------------------------------------------------
    # 使用訓練集進行五種特徵選擇分析，避免測試資料影響特徵排序。
    feature_selection = analyze_feature_selection(x_train, y_train)
    print("\n=== CRISP-DM 3.11: Feature Selection Analysis ===")
    print(feature_selection.round(4).to_string(index=False))

    # -------------------------------------------------
    # CRISP-DM Step 5: Model Evaluation
    # -------------------------------------------------
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

    # 額外比較專家指定特徵順序在五折交叉驗證下的累加效益。
    feature_count_comparison = create_feature_count_comparison(data)
    print("\n=== Feature Count Comparison ===")
    print(feature_count_comparison.round(6).to_string(index=False))

    print("\n=== CRISP-DM 4 & 5: Modeling and Evaluation ===")
    # 建立並訓練最終完整 Pipeline，再於保留的測試集上計算 RMSE 與 R-squared。
    model = build_pipeline()
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    rmse = float(mean_squared_error(y_test, predictions) ** 0.5)
    r2 = float(r2_score(y_test, predictions))
    print(f"RMSE: ${rmse:,.2f}")
    print(f"R-Squared (R2): {r2:.4f}")

    # -------------------------------------------------
    # CRISP-DM Step 6: Deployment
    # -------------------------------------------------
    # 儲存模型與分析結果。joblib 檔可直接載入預測，JSON 則記錄關鍵訓練資訊。
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


# 僅在直接執行此檔案時啟動完整流程；若被其他模組匯入，則不會自動訓練模型。
if __name__ == "__main__":
    main()

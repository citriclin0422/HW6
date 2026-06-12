"""Streamlit deployment for the interactive 50 Startups ML dashboard."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
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
DATA_FILE = BASE_DIR / "50_Startups.csv"
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
MODELS = ["Linear Regression", "Random Forest", "XGBoost"]
STATES = ["California", "Florida", "New York"]


st.set_page_config(
    page_title="50 Startups ML Lab",
    page_icon="📈",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp { background: #f5f2ea; color: #18212f; }
    [data-testid="stSidebar"] { background: #fffdf8; border-right: 1px solid #d9d4c8; }
    [data-testid="stMetric"] {
        background: #fffdf8;
        border: 1px solid #d9d4c8;
        border-radius: 16px;
        padding: 16px;
        box-shadow: 0 10px 28px rgba(33, 40, 52, .07);
    }
    div[data-testid="stDataFrame"] {
        border: 1px solid #d9d4c8;
        border-radius: 14px;
        overflow: hidden;
    }
    .hero {
        background: linear-gradient(135deg, #18212f, #2855d9);
        color: white;
        padding: 28px 32px;
        border-radius: 20px;
        margin-bottom: 22px;
    }
    .hero-kicker {
        color: #b8c9ff;
        font-size: 12px;
        font-weight: 800;
        letter-spacing: .16em;
        text-transform: uppercase;
    }
    .hero h1 { margin: 8px 0; font-size: 42px; }
    .hero p { margin: 0; color: #dce4ff; max-width: 820px; }
    .prediction-card {
        background: #18212f;
        color: white;
        padding: 22px;
        border-radius: 16px;
        margin-top: 8px;
    }
    .prediction-card small { color: #b9c4d5; font-weight: 700; }
    .prediction-card strong { display: block; font-size: 38px; margin-top: 5px; }
    </style>
    """,
    unsafe_allow_html=True,
)


def clean_feature_name(feature: str) -> str:
    return feature.replace("numeric__", "").replace("state__", "")


def create_regressor(
    model_name: str,
    random_state: int,
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
):
    if model_name == "Random Forest":
        return RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
        )
    if model_name == "XGBoost":
        return XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            n_jobs=1,
        )
    return LinearRegression()


@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_FILE)


@st.cache_data(show_spinner=False)
def run_experiment(
    model_name: str,
    algorithm: str,
    feature_count: int,
    test_size: float,
    random_state: int,
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
    rd_spend: float,
    administration: float,
    marketing_spend: float,
    state: str,
) -> dict[str, object]:
    data = load_data()
    features, target = data[RAW_FEATURES], data[TARGET]
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
    )

    feature_selection = analyze_feature_selection(x_train, y_train)
    rankings = build_feature_rankings(feature_selection)
    selected = rankings[algorithm][:feature_count]

    preprocessor = build_pipeline().named_steps["preprocessor"]
    transformed_train = preprocessor.fit_transform(x_train)
    transformed_test = preprocessor.transform(x_test)
    feature_names = preprocessor.get_feature_names_out()
    train_frame = pd.DataFrame(transformed_train, columns=feature_names)
    test_frame = pd.DataFrame(transformed_test, columns=feature_names)

    regressor = create_regressor(
        model_name,
        random_state,
        n_estimators,
        max_depth,
        learning_rate,
    )
    regressor.fit(train_frame[selected], y_train)
    predictions = regressor.predict(test_frame[selected])

    input_row = pd.DataFrame(
        [
            {
                "R&D Spend": rd_spend,
                "Administration": administration,
                "Marketing Spend": marketing_spend,
                "State": state,
            }
        ]
    )
    transformed_input = pd.DataFrame(
        preprocessor.transform(input_row), columns=feature_names
    )
    predicted_profit = float(regressor.predict(transformed_input[selected])[0])

    ranking_frame = pd.DataFrame(
        {
            f"Algorithm-{index} ({name})": [
                clean_feature_name(feature) for feature in ranked_features
            ]
            for index, (name, ranked_features) in enumerate(rankings.items(), start=1)
        },
        index=[f"Rank {rank}" for rank in range(1, len(feature_names) + 1)],
    )
    ranking_frame.index.name = "Rank"

    prediction_frame = pd.DataFrame(
        {"Actual Profit": y_test.to_numpy(), "Predicted Profit": predictions}
    )
    return {
        "rmse": float(mean_squared_error(y_test, predictions) ** 0.5),
        "r_squared": float(r2_score(y_test, predictions)),
        "training_rows": len(x_train),
        "test_rows": len(x_test),
        "selected_features": [clean_feature_name(name) for name in selected],
        "predicted_profit": predicted_profit,
        "prediction_frame": prediction_frame,
        "ranking_frame": ranking_frame,
        "feature_selection": feature_selection,
    }


if not DATA_FILE.exists():
    st.error("找不到 50_Startups.csv，請確認資料檔已上傳至 repository 根目錄。")
    st.stop()


st.markdown(
    """
    <div class="hero">
      <div class="hero-kicker">CRISP-DM Interactive Workbench</div>
      <h1>50 Startups ML Lab</h1>
      <p>互動比較三種模型、十種特徵選擇演算法與資料切分設定，並使用目前實驗組合預測新創公司 Profit。</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("實驗控制台")
    with st.form("experiment_form"):
        st.subheader("模型與特徵選擇")
        model_name = st.selectbox("模型", MODELS)
        algorithm = st.selectbox("特徵選擇演算法", ALGORITHMS)
        feature_count = st.slider("Top-K 特徵數", 1, 5, 1)
        test_size_percent = st.slider("測試集比例", 10, 40, 20, step=5)
        random_state = st.number_input(
            "Random State", min_value=0, max_value=10000, value=42
        )

        st.subheader("樹模型參數")
        n_estimators = st.number_input(
            "Estimators", min_value=50, max_value=1000, value=300, step=50
        )
        max_depth = st.slider("Max Depth", 1, 12, 3)
        learning_rate = st.slider(
            "Learning Rate", 0.01, 0.50, 0.05, step=0.01
        )

        st.subheader("新創公司預測輸入")
        rd_spend = st.number_input("R&D Spend", min_value=0.0, value=100000.0)
        administration = st.number_input(
            "Administration", min_value=0.0, value=120000.0
        )
        marketing_spend = st.number_input(
            "Marketing Spend", min_value=0.0, value=250000.0
        )
        state = st.selectbox("State", STATES)
        submitted = st.form_submit_button(
            "重新分析並預測", type="primary", width="stretch"
        )

parameters = (
    model_name,
    algorithm,
    feature_count,
    test_size_percent / 100,
    int(random_state),
    int(n_estimators),
    max_depth,
    learning_rate,
    rd_spend,
    administration,
    marketing_spend,
    state,
)

if submitted or "experiment_result" not in st.session_state:
    with st.spinner("正在重新排序、訓練與評估十種特徵選擇方法..."):
        st.session_state.experiment_result = run_experiment(*parameters)
        st.session_state.experiment_parameters = parameters

result = st.session_state.experiment_result
active_parameters = st.session_state.experiment_parameters
active_model, active_algorithm = active_parameters[0], active_parameters[1]

metric_columns = st.columns(4)
metric_columns[0].metric("Test RMSE", f"${result['rmse']:,.2f}")
metric_columns[1].metric("Test R-squared", f"{result['r_squared']:.4f}")
metric_columns[2].metric("Training Rows", result["training_rows"])
metric_columns[3].metric("Test Rows", result["test_rows"])

chart_column, prediction_column = st.columns([1.45, 0.85])
with chart_column:
    st.subheader("實際 Profit vs 預測 Profit")
    prediction_frame = result["prediction_frame"]
    minimum = prediction_frame.min().min()
    maximum = prediction_frame.max().max()
    figure, axis = plt.subplots(figsize=(7.5, 4.8))
    axis.scatter(
        prediction_frame["Actual Profit"],
        prediction_frame["Predicted Profit"],
        color="#2855d9",
        alpha=0.82,
        s=62,
        edgecolors="white",
        linewidths=0.8,
    )
    axis.plot(
        [minimum, maximum],
        [minimum, maximum],
        color="#ee7b3b",
        linestyle="--",
        label="Perfect prediction",
    )
    axis.set(
        xlabel="Actual Profit",
        ylabel="Predicted Profit",
        title=f"{active_model} · {active_algorithm}",
    )
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()
    st.pyplot(figure, width="stretch")
    plt.close(figure)

with prediction_column:
    st.subheader("目前實驗")
    st.caption(f"{active_model} · {active_algorithm}")
    st.write("**選取特徵：**")
    for index, feature in enumerate(result["selected_features"], start=1):
        st.markdown(f"`{index}. {feature}`")
    st.markdown(
        f"""
        <div class="prediction-card">
          <small>預測 Profit</small>
          <strong>${result["predicted_profit"]:,.2f}</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.subheader("10 種演算法特徵排名")
st.caption("排名會依目前 Test Size 與 Random State，在訓練資料上重新計算。")
st.dataframe(result["ranking_frame"], width="stretch")

with st.expander("查看特徵選擇原始分數"):
    st.dataframe(result["feature_selection"], width="stretch")

with st.expander("查看測試集實際值與預測值"):
    st.dataframe(
        result["prediction_frame"].style.format("${:,.2f}"),
        width="stretch",
    )

st.info(
    "此工具用於教學與探索。資料僅有 50 筆，預測結果可能對資料切分敏感，"
    "且特徵相關性不代表因果關係。"
)

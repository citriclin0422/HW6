# HW6：50 Startups 獲利預測

本專案使用 Kaggle `50_Startups.csv` 資料集，依照 **CRISP-DM** 流程建立新創公司獲利預測模型。流程包含資料理解、資料前處理、10 種特徵選擇演算法比較、線性迴歸模型評估，以及 FastAPI 模型部署。

## 專案目標

使用下列特徵預測新創公司的 `Profit`：

- `R&D Spend`
- `Administration`
- `Marketing Spend`
- `State`

## 10 種特徵選擇演算法

本專案比較以下方法：

1. Correlation
2. Mutual Information
3. Chi-Square
4. ANOVA F-Test
5. SelectKBest
6. RFE
7. SFS
8. Lasso
9. Random Forest
10. XGBoost

資料經標準化及 One-Hot Encoding 後，共有 5 個可排名特徵。圖表比較每種演算法依序加入 Rank 1 至 Rank 5 特徵後的 Test RMSE 與 Test R-squared。

## 特徵選擇效能比較

![Feature selection performance all-in-one](artifacts/feature_selection_performance_allinone.png)

## 10 種演算法排名比較

下方表格以 `Rank` 為列，並以 `Algorithm-1 (Correlation)` 至 `Algorithm-10 (XGBoost)` 為欄，顯示各演算法產生的特徵排名。

![Top 10 feature selection algorithms comparison](artifacts/feature_selection_10_algorithms_comparison.png)

## 最佳結果

10 種方法的 Rank 1 都是 `R&D Spend`。僅使用此特徵時，各方法得到相同的最佳測試集表現：

| 指標 | 結果 |
|---|---:|
| Test RMSE | **7,714.33** |
| Test R-squared | **0.9265** |
| 最佳特徵 | **R&D Spend** |

若需選擇單一方法，Correlation 計算速度快、結果容易解釋，且在本資料集取得相同最佳效能。

## 執行方式

安裝相依套件：

```powershell
pip install -r requirements.txt
```

訓練模型並重新產生圖表：

```powershell
python .\train_linear_regression.py
```

啟動 FastAPI：

```powershell
uvicorn app:app --reload
```

API 文件：

```text
http://127.0.0.1:8000/docs
```

## 主要輸出

| 檔案 | 說明 |
|---|---|
| `train_linear_regression.py` | CRISP-DM 訓練、評估及輸出流程 |
| `artifacts/feature_selection_performance_allinone.csv` | 10 種演算法、不同特徵數量的完整評估結果 |
| `artifacts/feature_selection_results.csv` | 各特徵選擇方法的分數與排名 |
| `artifacts/feature_selection_performance_allinone.png` | Test RMSE、R-squared 與排名綜合圖表 |
| `artifacts/feature_selection_10_algorithms_comparison.png` | 10 種演算法比較圖表 |
| `artifacts/best_model.joblib` | FastAPI 可載入的訓練模型 |

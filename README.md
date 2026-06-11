# HW6：新創公司利潤預測與特徵選擇

本專案使用 Kaggle `50_Startups.csv` 資料集，依循 **CRISP-DM** 流程完成新創公司利潤預測、特徵選擇、模型評估與 FastAPI 部署。

## 簡報與動畫影片

- [專案簡報：Startup_Profit_Prediction.pptx](Startup_Profit_Prediction.pptx)
- [動畫解說影片：Startup_Profit_Prediction_Animated.mp4](Startup_Profit_Prediction_Animated.mp4)

動畫影片長度約 2 分 30 秒，包含 15 個動畫場景與繁體中文女性旁白，內容涵蓋完整的分析與部署流程。

## 專案目標

透過以下特徵預測新創公司的利潤 `Profit`：

- `R&D Spend`：研發支出
- `Administration`：行政支出
- `Marketing Spend`：行銷支出
- `State`：公司所在州別

## 主要發現

| 特徵 | 與利潤的相關係數 |
|---|---:|
| `R&D Spend` | 0.9729 |
| `Marketing Spend` | 0.7478 |
| `Administration` | 0.2007 |

- 研發支出是最重要且最穩定的利潤預測指標。
- 行銷支出能提供額外的泛化預測能力。
- 行政支出與州別的額外預測能力有限。
- 由於資料集只有 50 筆，模型應作為決策支援工具，不宜用於因果推論。

## 模型評估

五折交叉驗證顯示，最佳特徵組合為：

- `R&D Spend`
- `Marketing Spend`

| 評估指標 | 結果 |
|---|---:|
| CV RMSE | **8,883.70** |
| CV R-squared | **0.9389** |
| 最終模型 Test RMSE | **9,055.96** |
| 最終模型 Test R-squared | **0.8987** |

最終部署模型採用多元線性迴歸，在小型資料集上具備較佳的穩定性與商業可解釋性。

## 執行方式

安裝相依套件：

```powershell
pip install -r requirements.txt
```

訓練模型：

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

## 專案檔案

| 檔案 | 說明 |
|---|---|
| `Startup_Profit_Prediction.pptx` | 專案成果簡報 |
| `Startup_Profit_Prediction_Animated.mp4` | 繁體中文動畫解說影片 |
| `train_linear_regression.py` | 模型訓練與評估 |
| `app.py` | FastAPI 預測服務 |
| `50_Startups.csv` | 原始資料集 |
| `HW6.md` | CRISP-DM 專案說明 |
| `artifacts/` | 圖表與分析輸出 |

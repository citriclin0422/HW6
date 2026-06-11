# HW6：新創公司利潤預測與特徵選擇

本專案使用 `50_Startups.csv` 資料集，依循 **CRISP-DM** 機器學習流程，
建立新創公司利潤預測模型。專案包含探索式資料分析、資料前處理、五種特徵
選擇演算法、模型評估、視覺化、模型儲存，以及 FastAPI 預測服務。

![CRISP-DM 專案流程](artifacts/workflow.png)

## 專案目標

根據下列資訊預測新創公司的 `Profit`：

- `R&D Spend`：研發支出
- `Administration`：行政支出
- `Marketing Spend`：行銷支出
- `State`：公司所在州別

主要研究問題：

- 研發支出是否為預測利潤最重要的特徵？
- 行銷支出是否能提升預測表現？
- 行政支出與公司所在地是否具有足夠的預測價值？
- 使用多少個特徵能獲得最佳泛化能力？

## 資料集概覽

| 項目 | 結果 |
|---|---:|
| 資料筆數 | 50 |
| 欄位數量 | 5 |
| 缺失值 | 0 |
| 重複資料 | 0 |

### 與利潤的相關係數

| 特徵 | 相關係數 |
|---|---:|
| `R&D Spend` | 0.9729 |
| `Marketing Spend` | 0.7478 |
| `Administration` | 0.2007 |

`R&D Spend` 與利潤具有最強的線性關係；`Marketing Spend` 次之；
`Administration` 的關係較弱。

## 資料前處理

- 使用 `StandardScaler` 標準化數值特徵。
- 使用 One-Hot Encoding 處理 `State`。
- 移除第一個州別類別，避免虛擬變數陷阱。
- 使用 80% 訓練資料與 20% 測試資料。
- 設定 `random_state=42`，確保結果可重現。
- 使用 Scikit-Learn `Pipeline` 避免資料洩漏。

## 五種特徵選擇演算法

本專案比較下列五種方法，並測試各方法選擇 1 至 5 個特徵時的模型表現：

1. **相關性分析（Correlation Analysis）**
2. **SelectKBest F-Regression**
3. **遞迴特徵消除（Recursive Feature Elimination, RFE）**
4. **Lasso Regression**
5. **Random Forest Feature Importance**

所有方法皆將 `R&D Spend` 排名為第一重要特徵，並將
`Marketing Spend` 排名為第二重要特徵。

![五種特徵選擇演算法效能比較](artifacts/feature_selection_performance_allinone.png)

## 主要結果

### 固定測試集結果

固定的 80/20 測試集中，只使用 `R&D Spend` 可獲得最佳表現：

| 指標 | 結果 |
|---|---:|
| RMSE | **7,714.33** |
| R-squared | **0.9265** |

由於資料集僅有 50 筆資料，此結果可能受到單次資料切分影響。

### 五折交叉驗證結果

使用打亂後的五折交叉驗證時，最佳組合為：

- `R&D Spend`
- `Marketing Spend`

| 指標 | 結果 |
|---|---:|
| CV RMSE | **8,883.70** |
| CV R-squared | **0.9389** |

加入州別特徵後，平均交叉驗證表現反而下降。相較於單次測試集結果，
交叉驗證對小型資料集通常較穩定，因此兩個特徵的結果具有較強參考價值。

![特徵數量交叉驗證比較](artifacts/final_output_diagram.gif)

### 部署模型結果

最終部署模型使用所有原始輸入特徵，並採用多元線性迴歸：

| 指標 | 結果 |
|---|---:|
| Test RMSE | 9,055.96 |
| Test R-squared | 0.8987 |

線性迴歸模型可解釋約 89.87% 的測試集利潤變異，具有良好的可解釋性，
適合此小型且主要呈線性關係的資料集。

## 重要結論

- `R&D Spend` 是最重要且最穩定的利潤預測特徵。
- `Marketing Spend` 能提供額外且有用的預測資訊。
- `Administration` 的預測能力有限。
- 控制支出後，`State` 對利潤的影響很小。
- 交叉驗證支持使用 `R&D Spend` 與 `Marketing Spend` 兩個特徵。
- 本模型用於決策支援，不應被解讀為因果推論。

## 專案結構

| 路徑 | 用途 |
|---|---|
| `train_linear_regression.py` | 完整訓練、評估與視覺化流程 |
| `50_Startups.csv` | 原始資料集 |
| `HW6.md` | 完整 CRISP-DM 專案報告 |
| `app.py` | FastAPI 預測服務 |
| `artifacts/workflow.png` | 專案流程圖 |
| `artifacts/feature_selection_performance_allinone.png` | 五種特徵選擇演算法比較圖 |
| `artifacts/final_output_diagram.gif` | 特徵數量交叉驗證比較圖 |
| `artifacts/` | 模型、指標、CSV 與其他視覺化產出 |
| `requirements.txt` | Python 套件需求 |

## 執行方式

安裝所需套件：

```powershell
pip install -r requirements.txt
```

執行訓練流程並重新產生所有圖表與模型：

```powershell
python .\train_linear_regression.py
```

啟動 FastAPI 預測服務：

```powershell
uvicorn app:app --reload
```

開啟互動式 API 文件：

```text
http://127.0.0.1:8000/docs
```

## API 使用方式

傳送預測請求至 `POST /predict`：

```json
{
  "rd_spend": 100000,
  "administration": 120000,
  "marketing_spend": 250000,
  "state": "New York"
}
```

## 完整文件

請閱讀 [HW6.md](HW6.md)，查看完整的 CRISP-DM 分析、特徵選擇結果、
模型比較、限制與部署說明。

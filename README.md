# Rossmann Store Sales — Time Series Analysis & Forecasting

**SARIMA modeling and deep learning forecasting comparison on retail sales data**

📍 University of Minnesota · Carlson School of Management · MSBA Program · MSBA 6431 Time Series Analysis and Forecasting

---

## Executive Summary

This project applies classical time series methodology to daily sales data from Store 262 of the Rossmann drugstore chain, following the standard Box-Jenkins procedure for SARIMA model building: transformation, stationarity testing, order identification via ACF/PACF/EACF, and AIC/BIC-based model selection. The resulting model is evaluated on a held-out test set and benchmarked against an LSTM forecasting model to compare statistical and deep learning approaches on the same forecasting task.

The final SARIMA model — ARIMA(2,0,3)(2,0,0)[6] — outperformed the LSTM on this dataset (test RMSE 4048.2 vs. 4712.4), reflecting the strong, regular weekly seasonality in retail sales that SARIMA is well-suited to capture, especially with a relatively small training set (932 observations).

**Key deliverables:**

- Stationarity testing and transformation of a real-world retail sales series
- SARIMA model identification and selection using ACF/PACF/EACF and AIC/BIC
- Coefficient significance testing and business interpretation of the final model
- Out-of-sample forecast evaluation with 80%/95% prediction intervals
- LSTM forecasting model as a deep learning benchmark, with a comparative analysis of accuracy, speed, and interpretability

---

## Results

| Model | Test RMSE | Notes |
|---|---|---|
| **SARIMA(2,0,3)(2,0,0)[6]** | **4048.2** | All coefficients statistically significant |
| LSTM | 4712.4 | Single-layer, 32 units, no hyperparameter tuning |

SARIMA outperformed the LSTM by roughly 14% on test RMSE. This is consistent with the data's strong, regular weekly seasonal structure — exactly the kind of pattern SARIMA is designed to capture — combined with a training set (932 observations) too small for the LSTM's added flexibility to pay off.

---

## Repository Structure

```text
.
├── notebooks/
│   ├── sarima_analysis.Rmd      # Full SARIMA analysis: stationarity, model selection, forecasting
│   └── lstm_forecast.py         # LSTM forecasting benchmark for comparison
├── lstm_forecast_plot.png       # LSTM forecast visualization
├── data/
│   └── README.md                # Dataset description & download instructions
└── README.md
```

---

## Dataset

**Rossmann Store Sales (Kaggle)**

Daily sales revenue (EUR) for Store 262, one of Rossmann's ~3,000 drugstores across Europe, covering 01/01/2013 to 07/31/2015 on days the store was open.

- Source: [https://www.kaggle.com/competitions/rossmann-store-sales/overview](https://www.kaggle.com/competitions/rossmann-store-sales/overview)
- Sample size: 942 daily observations (last 10 held out for testing)

Raw data is not included in this repository. See [`data/README.md`](data/README.md) for download instructions.

> **Note:** This project uses only `train.csv`, splitting the last 10 observations off as a held-out test set for model evaluation. It is a course time series analysis exercise, not a submission to the Kaggle competition leaderboard, and does not produce a Kaggle-format `submission.csv`.

---

## Methodology

### 1. Data Preparation & Train-Test Split

Store 262's sales are filtered to open days only and converted to a time series object with weekly frequency (6, matching the store's 6-day operating week). The last 10 observations are held out for testing; the rest form the training set.

### 2. SARIMA Model Building

Following the Box-Jenkins procedure:

- **Transformation** — log transform applied to stabilize variance
- **Stationarity (ADF test)** — the log-transformed series is confirmed stationary, so no differencing is required
- **Order identification** — ACF, PACF, and EACF are examined to identify candidate ARMA orders, with a seasonal component at lag 6
- **Model selection** — several candidate models are compared by AIC/BIC alongside `auto.arima()`'s suggestion; `auto.arima()`'s ARIMA(2,0,3)(2,0,0)[6] outperforms all manually specified candidates with every coefficient statistically significant

**Final model:**

$$Y_t (\log(\text{Sales}_t)) = 9.9161 - 0.3482(Y_{t-1}-9.9161) - 0.8795(Y_{t-2}-9.9161) + e_t + 0.5717e_{t-1} + 1.0089e_{t-2} + 0.4077e_{t-3}$$

The non-seasonal AR(2)/MA(3) component captures short-term carryover dynamics (e.g., lingering effects of a one-day promotion), while the seasonal AR(2) term at lag 6 reflects the store's stable weekly shopping pattern.

**Business interpretation:** The AR(2) component supports short-term inventory adjustments — a recent upward trend signals proactive restocking. The seasonal AR(2) term suggests staffing and restocking schedules benefit from a two-week lookback, since sales two weeks prior continue to inform expected demand.

### 3. Forecast Evaluation

The fitted model forecasts the 10-day test period, with results converted back to the original EUR scale. Test RMSE: **4048.2**. Day-by-day percentage errors are mostly in the 4–16% range, with one outlier day (31.5% error) attributable to a demand spike the model could not anticipate. 80%/95% prediction intervals reasonably cover most actual test values.

### 4. LSTM Benchmark

A single-layer LSTM (32 units) is trained on the same log-transformed series using an 18-step lookback window (3 seasonal cycles) to forecast the same 10-day horizon, enabling direct comparison with the SARIMA model.

![LSTM Forecast vs Actual Sales](lstm_forecast_plot.png)

**Test RMSE: 4712.4** — approximately 14% worse than SARIMA.

**Why SARIMA outperformed LSTM here:**

1. The training set (932 observations) is relatively small for deep learning models, which typically need larger datasets to learn robust patterns
2. The LSTM used a lightweight, single-layer architecture without hyperparameter tuning
3. The data's clear, regular weekly seasonality is exactly the structure SARIMA is designed to capture, leaving less room for LSTM's added flexibility to provide an advantage

Given the interpretability requirement and the strong seasonal structure of retail sales data, SARIMA is the more suitable choice for this business context — it forecasts more accurately here and produces coefficients that translate directly into actionable inventory and staffing decisions.

---

## Setup

### R (SARIMA analysis)

```r
install.packages(c("TSA", "ggplot2", "dplyr", "forecast", "tseries", "Metrics"))
```

Open `notebooks/sarima_analysis.Rmd` in RStudio and run all chunks. Ensure `data/train.csv` is present (see [`data/README.md`](data/README.md)).

### Python (LSTM benchmark)

```bash
pip install numpy pandas tensorflow matplotlib
python notebooks/lstm_forecast.py
```

---

## Tools & Technologies

| Layer | Technology |
|---|---|
| Statistical modeling | R · TSA · forecast · tseries |
| Deep learning | Python · TensorFlow/Keras (LSTM) |
| Visualization | ggplot2 · Matplotlib |

---

## Usage and License Note

This repository is shared for academic and portfolio purposes. Please contact the author before reusing or redistributing the code.

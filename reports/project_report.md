# Project Report: CIPLA Stock Price Prediction System

**Subject:** Artificial Intelligence and Machine Learning  
**Algorithm:** XGBoost Regression  
**Dataset:** CIPLA NSE Historical Data (2000–2024)

---

## 1. Problem Statement

Predict the closing price of CIPLA stock using historical OHLCV data and engineered time-series features. This is a supervised regression problem with temporal dependency.

## 2. Dataset Description

- **Source:** NSE India — CIPLA equity
- **Records:** ~5,306 trading days
- **Features:** Date, Symbol, Open, High, Low, Close, Volume, Prev Close, VWAP, Turnover, Deliverable Volume
- **Target:** `Close` price (continuous)

## 3. Preprocessing

1. Dropped low-value columns: Symbol, Series, VWAP, Turnover, Trades, Deliverable Volume, %Deliverble
2. Parsed Date column and sorted chronologically
3. Forward-filled missing values; dropped remaining NaN rows
4. MinMaxScaler applied to numeric columns before EDA analysis

## 4. Exploratory Data Analysis

Key findings:
- CIPLA stock grew from ~₹165 (2000) to ~₹1,465 (2024)
- Strong upward trend from 2020 onward
- Volume shows high spikes during market events
- Close, Open, High, Low, Prev Close are highly correlated (>0.99)

## 5. Feature Engineering

| Feature Group | Features | Rationale |
|---|---|---|
| Moving Averages | MA5, MA10, MA20, MA50 | Trend identification |
| Exponential MA | EMA12, EMA26, MACD | Momentum indicator |
| Lag Features | Close_lag1..lag10 | Temporal autocorrelation |
| RSI | RSI14 | Overbought/oversold |
| Volatility | Volatility_10 | Risk measure |
| Derived | Daily_Return, Price_Range, Open_Close_Diff | Intraday movement |
| Date Parts | Year, Month, DayOfWeek | Seasonal patterns |

## 6. Model: XGBoost Regressor

**Hyperparameters:**
- `n_estimators`: 500
- `learning_rate`: 0.05
- `max_depth`: 6
- `subsample`: 0.8
- `colsample_bytree`: 0.8

**Why XGBoost:**
- Ensemble gradient boosting handles non-linearity in financial data
- Regularization (L1/L2) prevents overfitting
- Handles missing values natively
- Feature importance for interpretability
- No assumption of stationarity required

## 7. Evaluation Metrics

| Metric | Formula | Interpretation |
|---|---|---|
| RMSE | √(mean(y−ŷ)²) | Penalises large errors |
| MAE | mean(|y−ŷ|) | Average absolute error (₹) |
| R² | 1 - SS_res/SS_tot | Proportion of variance explained |
| MAPE | mean(|y−ŷ|/y)×100 | Percentage error |

## 8. Results

The XGBoost model achieves R² > 0.98 on the test set, demonstrating strong predictive accuracy. `Close_lag1` (previous day's close) is the dominant feature, followed by EMA and MA features.

## 9. Flask Web Application

The application provides a 5-step interactive interface:
1. Upload data and view preprocessing results
2. Explore EDA visualizations
3. View engineered features
4. Train the XGBoost model
5. View predictions and evaluation metrics

## 10. Conclusion

This project successfully implements an end-to-end ML pipeline for stock price prediction. XGBoost with temporal features achieves near-perfect tracking of actual prices, validating the effectiveness of lag and technical indicator features for time-series regression.

**Limitations:** Past performance does not guarantee future results. The model does not account for fundamentals, news sentiment, or macroeconomic factors.

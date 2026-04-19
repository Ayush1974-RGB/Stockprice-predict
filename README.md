# 📈Stock Price Prediction System

End-to-end stock price prediction using **XGBoost Regression** with a **Flask** web application. Built for academic lab evaluation demonstrating Supervised Learning and Time-Series Forecasting.

---

## 🗂 Project Structure

```
Stock-Price-Prediction-System/
├── data/
│   ├── raw/CIPLA.csv              # Original NSE dataset
│   └── processed/                 # Cleaned & feature-engineered CSVs
├── src/
│   ├── preprocessing.py           # Data cleaning, normalization
│   ├── eda.py                     # EDA visualizations
│   ├── feature_engineering.py     # MA, lag, RSI, MACD, etc.
│   ├── train.py                   # XGBoost model training
│   └── evaluate.py                # RMSE, MAE, R², plots
├── models/
│   └── xgboost_model.pkl          # Saved trained model
├── app/
│   ├── app.py                     # Flask backend
│   ├── templates/index.html       # Frontend UI
│   └── static/                    # CSS + JS
├── visuals/                       # Generated plot images
├── reports/project_report.md
├── requirements.txt
└── main.py                        # CLI pipeline runner
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the full pipeline (pre-trains and saves model)
```bash
python main.py
```

### 3. Launch the Flask web app
```bash
cd app
python app.py
```
Open **http://localhost:5000** in your browser.

---

## 🔬 ML Pipeline

| Step | Module | Description |
|------|--------|-------------|
| 1 | `preprocessing.py` | Drop irrelevant columns, forward-fill, normalize |
| 2 | `eda.py` | Price history, OHLC, volume, correlation heatmap |
| 3 | `feature_engineering.py` | MA (5/10/20/50), EMA (12/26), MACD, RSI-14, lag features, volatility |
| 4 | `train.py` | XGBoost Regressor (500 trees, lr=0.05, depth=6), 80/20 chronological split |
| 5 | `evaluate.py` | RMSE, MAE, R², MAPE — actual vs predicted plots |

---

## 📊 Dataset

**CIPLA.NSE** — National Stock Exchange historical data (2000–2024)
- ~5,300 trading days
- Features: Date, Open, High, Low, Close, Volume, Prev Close

---

## 🧠 Why XGBoost?

- Handles non-linear relationships in financial time series
- Built-in regularization prevents overfitting
- Feature importance interpretability
- Faster training vs. deep learning models for tabular data
- Consistently outperforms Linear Regression / SVR on this dataset

---

## 📁 Academic Use

This project demonstrates:
- **Supervised Learning** — regression on labelled stock price data
- **Time-Series Forecasting** — lag features, rolling statistics, chronological split
- **Feature Engineering** — domain-specific technical indicators (MACD, RSI, Bollinger-style MAs)
- **Model Evaluation** — RMSE, MAE, R², MAPE
- **Full-Stack ML** — Python backend + interactive Flask UI

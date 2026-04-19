"""
evaluate.py
Model evaluation: RMSE, MAE, R², and visualisations.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import pickle

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

FEATURED_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'featured_data.csv')
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'xgboost_model.pkl')
SCALER_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'scaler.pkl')
VISUALS_DIR = os.path.join(os.path.dirname(__file__), '..', 'visuals')

FEATURE_COLS = [
    'Open', 'High', 'Low', 'Volume', 'Prev Close',
    'MA5', 'MA10', 'MA20', 'MA50',
    'Close_lag1', 'Close_lag2', 'Close_lag3', 'Close_lag5', 'Close_lag10',
    'Daily_Return', 'Price_Range', 'Open_Close_Diff',
    'EMA12', 'EMA26', 'MACD', 'Volatility_10', 'RSI14',
    'Year', 'Month', 'DayOfWeek'
]


def _save(fig, name):
    os.makedirs(VISUALS_DIR, exist_ok=True)
    path = os.path.join(VISUALS_DIR, name)
    fig.savefig(path, bbox_inches='tight', dpi=120)
    plt.close(fig)
    return path


def evaluate(model=None, scaler=None, df=None) -> dict:
    if df is None:
        df = pd.read_csv(FEATURED_PATH, parse_dates=['Date'])
    if model is None:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
    if scaler is None:
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)

    features = [c for c in FEATURE_COLS if c in df.columns]
    X = df[features].values
    y = df['Close'].values
    dates = df['Date'].values

    split = int(len(X) * 0.80)
    X_test = scaler.transform(X[split:])
    y_test = y[split:]
    dates_test = dates[split:]

    y_pred = model.predict(X_test)

    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))
    mape = float(np.mean(np.abs((y_test - y_pred) / (y_test + 1e-9))) * 100)

    # ── Plot 1: Actual vs Predicted ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(dates_test, y_test, label='Actual', color='#2563EB', linewidth=1.2)
    ax.plot(dates_test, y_pred, label='Predicted', color='#F97316', linewidth=1.2, linestyle='--')
    ax.set_title('Actual vs Predicted Close Price (Test Set)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price (₹)')
    ax.legend()
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    pred_plot = _save(fig, 'predictions.png')

    # ── Plot 2: Residuals ─────────────────────────────────────────────────────
    residuals = y_test - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].scatter(y_pred, residuals, alpha=0.4, color='#7C3AED', s=8)
    axes[0].axhline(0, color='red', linestyle='--')
    axes[0].set_title('Residuals vs Predicted')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Residual')
    axes[0].grid(alpha=0.3)

    axes[1].hist(residuals, bins=50, color='#0891B2', edgecolor='white')
    axes[1].set_title('Residuals Distribution')
    axes[1].set_xlabel('Residual (₹)')
    axes[1].set_ylabel('Count')
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    residual_plot = _save(fig, 'residuals.png')

    # ── Plot 3: Feature Importance ────────────────────────────────────────────
    importance = dict(zip(features, model.feature_importances_))
    top = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:12]
    labels, vals = zip(*top)
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(labels[::-1], vals[::-1], color='#2563EB')
    ax.set_title('Top-12 Feature Importances (XGBoost)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Importance Score')
    ax.grid(alpha=0.3, axis='x')
    plt.tight_layout()
    importance_plot = _save(fig, 'feature_importance.png')

    report = {
        'rmse': round(rmse, 4),
        'mae': round(mae, 4),
        'r2': round(r2, 4),
        'mape': round(mape, 4),
        'test_size': len(y_test),
        'pred_plot': pred_plot,
        'residual_plot': residual_plot,
        'importance_plot': importance_plot,
        'predictions': [
            {'date': str(pd.Timestamp(d).date()), 'actual': round(float(a), 2), 'predicted': round(float(p), 2)}
            for d, a, p in zip(dates_test[-30:], y_test[-30:], y_pred[-30:])
        ]
    }
    return report


if __name__ == '__main__':
    r = evaluate()
    print(f"RMSE : ₹{r['rmse']}")
    print(f"MAE  : ₹{r['mae']}")
    print(f"R²   : {r['r2']}")
    print(f"MAPE : {r['mape']}%")

"""
train.py
Model training using XGBoost Regression.
"""

import pandas as pd
import numpy as np
import os
import pickle
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
try:
    import xgboost as xgb
    _USE_XGB = True
except ImportError:
    from sklearn.ensemble import GradientBoostingRegressor as _GBR
    _USE_XGB = False

FEATURED_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'featured_data.csv')
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'xgboost_model.pkl')
SCALER_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'scaler.pkl')
TRAIN_META_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'train_meta.json')

FEATURE_COLS = [
    'Open', 'High', 'Low', 'Volume', 'Prev Close',
    'MA5', 'MA10', 'MA20', 'MA50',
    'Close_lag1', 'Close_lag2', 'Close_lag3', 'Close_lag5', 'Close_lag10',
    'Daily_Return', 'Price_Range', 'Open_Close_Diff',
    'EMA12', 'EMA26', 'MACD', 'Volatility_10', 'RSI14',
    'Year', 'Month', 'DayOfWeek'
]
TARGET_COL = 'Close'


def train(df: pd.DataFrame = None) -> dict:
    if df is None:
        df = pd.read_csv(FEATURED_PATH, parse_dates=['Date'])

    features = [c for c in FEATURE_COLS if c in df.columns]
    X = df[features].values
    y = df[TARGET_COL].values

    # Train / test split (80/20 chronological)
    split = int(len(X) * 0.80)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    dates_test = df['Date'].iloc[split:].values

    # Scale features
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # XGBoost model (falls back to sklearn GradientBoostingRegressor if xgboost not installed)
    if _USE_XGB:
        model = xgb.XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='reg:squarederror',
            random_state=42,
            verbosity=0,
        )
        model.fit(
            X_train_s, y_train,
            eval_set=[(X_test_s, y_test)],
            verbose=False,
        )
    else:
        model = _GBR(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            random_state=42,
            verbose=0,
        )
        model.fit(X_train_s, y_train)

    # Save artefacts
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)

    # Feature importance (top 10) – works for both XGBoost and sklearn GB
    importances = model.feature_importances_
    importance = dict(zip(features, importances.tolist()))
    top10 = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]

    model_name = 'XGBoostRegressor' if _USE_XGB else 'GradientBoostingRegressor (sklearn fallback)'

    meta = {
        'features': features,
        'train_size': int(split),
        'test_size': int(len(X_test)),
        'n_estimators': 500 if _USE_XGB else 300,
        'learning_rate': 0.05,
        'max_depth': 6,
        'model_name': model_name,
        'feature_importance': top10,
    }
    with open(TRAIN_META_PATH, 'w') as f:
        json.dump(meta, f)

    return model, scaler, X_test_s, y_test, dates_test, meta


def load_model():
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    with open(TRAIN_META_PATH) as f:
        meta = json.load(f)
    return model, scaler, meta


if __name__ == '__main__':
    model, scaler, X_test_s, y_test, dates_test, meta = train()
    print(f"Training complete. Model saved to {MODEL_PATH}")
    print(f"Train: {meta['train_size']} | Test: {meta['test_size']}")
    print("Top-5 features:", meta['feature_importance'][:5])

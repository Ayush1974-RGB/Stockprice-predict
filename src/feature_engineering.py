"""
feature_engineering.py
Lag features, moving averages, and time-series derived features.
"""

import pandas as pd
import numpy as np
import os

PROCESSED_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'processed_data.csv')
FEATURED_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'featured_data.csv')


def engineer_features(df: pd.DataFrame = None) -> tuple:
    if df is None:
        df = pd.read_csv(PROCESSED_PATH, parse_dates=['Date'])

    df = df.sort_values('Date').reset_index(drop=True)
    report = {}

    # ── 1. Moving Averages ─────────────────────────────────────────────────────
    for w in [5, 10, 20, 50]:
        df[f'MA{w}'] = df['Close'].rolling(w).mean()
    report['moving_averages'] = ['MA5', 'MA10', 'MA20', 'MA50']

    # ── 2. Lag Features ────────────────────────────────────────────────────────
    for lag in [1, 2, 3, 5, 10]:
        df[f'Close_lag{lag}'] = df['Close'].shift(lag)
    report['lag_features'] = ['Close_lag1', 'Close_lag2', 'Close_lag3', 'Close_lag5', 'Close_lag10']

    # ── 3. Price differences ───────────────────────────────────────────────────
    df['Daily_Return'] = df['Close'].pct_change()
    df['Price_Range'] = df['High'] - df['Low']
    df['Open_Close_Diff'] = df['Close'] - df['Open']
    report['derived_features'] = ['Daily_Return', 'Price_Range', 'Open_Close_Diff']

    # ── 4. Exponential Moving Average ─────────────────────────────────────────
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    report['ema_features'] = ['EMA12', 'EMA26', 'MACD']

    # ── 5. Volatility ──────────────────────────────────────────────────────────
    df['Volatility_10'] = df['Daily_Return'].rolling(10).std()
    report['volatility_features'] = ['Volatility_10']

    # ── 6. RSI (14-period) ────────────────────────────────────────────────────
    delta = df['Close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df['RSI14'] = 100 - (100 / (1 + rs))
    report['rsi_features'] = ['RSI14']

    # ── 7. Date parts ──────────────────────────────────────────────────────────
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    report['date_features'] = ['Year', 'Month', 'DayOfWeek']

    # ── 8. Drop NaN rows introduced by rolling ──────────────────────────────
    df = df.dropna().reset_index(drop=True)
    report['shape_after'] = list(df.shape)
    report['total_features'] = len(df.columns) - 1  # minus Date

    df.to_csv(FEATURED_PATH, index=False)
    report['featured_path'] = FEATURED_PATH
    report['sample'] = df.tail(3).to_dict(orient='records')

    return df, report


if __name__ == '__main__':
    df, r = engineer_features()
    print(f"Feature engineering done. Shape: {r['shape_after']}")
    print(f"Features created: {r['total_features']}")

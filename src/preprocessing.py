"""
preprocessing.py
Data cleaning and preparation for CIPLA stock price prediction.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
import json

RAW_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'CIPLA.csv')
PROCESSED_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'processed_data.csv')


def load_data(path: str = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def preprocess(df: pd.DataFrame) -> dict:
    report = {}

    # ── 1. Shape before cleaning ──────────────────────────────────────────────
    report['shape_before'] = list(df.shape)
    report['columns'] = list(df.columns)
    report['missing_before'] = df.isnull().sum().to_dict()

    # ── 2. Drop columns with >40 % missing or irrelevant ──────────────────────
    drop_cols = [c for c in ['Trades', 'Deliverable Volume', '%Deliverble', 'Symbol', 'Series', 'VWAP', 'Turnover', 'Last']
                 if c in df.columns]
    df = df.drop(columns=drop_cols)
    report['dropped_columns'] = drop_cols

    # ── 3. Parse Date ─────────────────────────────────────────────────────────
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    # ── 4. Fill remaining missing values (forward fill then drop) ─────────────
    df = df.ffill().dropna()
    report['missing_after'] = df.isnull().sum().to_dict()
    report['shape_after'] = list(df.shape)

    # ── 5. Normalise numeric columns (store scaler params) ────────────────────
    numeric_cols = ['Prev Close', 'Open', 'High', 'Low', 'Close', 'Volume']
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[numeric_cols])
    df_scaled = df.copy()
    df_scaled[numeric_cols] = scaled

    # Save scaler ranges for reference
    report['scaler_min'] = dict(zip(numeric_cols, scaler.data_min_.tolist()))
    report['scaler_max'] = dict(zip(numeric_cols, scaler.data_max_.tolist()))

    # Save processed (un-scaled) data for feature engineering
    os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)
    report['processed_path'] = PROCESSED_PATH

    # Sample rows for UI display
    report['sample_rows'] = df.head(5).to_dict(orient='records')
    report['dtypes'] = {col: str(dtype) for col, dtype in df.dtypes.items()}

    return df, report


def run():
    df_raw = load_data()
    df_clean, report = preprocess(df_raw)
    print("Preprocessing complete.")
    print(f"  Shape: {report['shape_before']} → {report['shape_after']}")
    print(f"  Dropped: {report['dropped_columns']}")
    return df_clean, report


if __name__ == '__main__':
    run()

"""
eda.py
Exploratory Data Analysis – statistical summaries and visualisations.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import json


PROCESSED_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'processed_data.csv')
VISUALS_DIR = os.path.join(os.path.dirname(__file__), '..', 'visuals')


def _save(fig, name):
    os.makedirs(VISUALS_DIR, exist_ok=True)
    path = os.path.join(VISUALS_DIR, name)
    fig.savefig(path, bbox_inches='tight', dpi=120)
    plt.close(fig)
    return path


def run_eda(df: pd.DataFrame = None) -> dict:
    if df is None:
        df = pd.read_csv(PROCESSED_PATH, parse_dates=['Date'])

    report = {}

    # ── 1. Statistical summary ────────────────────────────────────────────────
    desc = df.describe().round(2)
    report['describe'] = json.loads(desc.to_json())

    # ── 2. Close price history ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df['Date'], df['Close'], color='#2563EB', linewidth=0.9)
    ax.set_title('CIPLA – Closing Price History', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price (₹)')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    fig.autofmt_xdate()
    ax.grid(alpha=0.3)
    report['close_history_plot'] = _save(fig, 'close_history.png')

    # ── 3. Volume over time ────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.bar(df['Date'], df['Volume'], color='#7C3AED', alpha=0.6, width=1)
    ax.set_title('CIPLA – Trading Volume', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Volume')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    fig.autofmt_xdate()
    ax.grid(alpha=0.3, axis='y')
    report['volume_plot'] = _save(fig, 'volume.png')

    # ── 4. OHLC candlestick-style box (last 60 trading days) ─────────────────
    last60 = df.tail(60).copy()
    fig, ax = plt.subplots(figsize=(12, 4))
    up = last60[last60['Close'] >= last60['Open']]
    dn = last60[last60['Close'] < last60['Open']]
    w = 0.6
    ax.bar(up['Date'], up['Close'] - up['Open'], w, bottom=up['Open'], color='#16A34A', label='Up')
    ax.bar(dn['Date'], dn['Close'] - dn['Open'], w, bottom=dn['Open'], color='#DC2626', label='Down')
    ax.vlines(up['Date'], up['Low'], up['High'], color='#16A34A', linewidth=0.8)
    ax.vlines(dn['Date'], dn['Low'], dn['High'], color='#DC2626', linewidth=0.8)
    ax.set_title('CIPLA – OHLC (Last 60 Days)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price (₹)')
    ax.legend()
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    report['ohlc_plot'] = _save(fig, 'ohlc.png')

    # ── 5. Correlation heatmap ─────────────────────────────────────────────────
    num_df = df.select_dtypes(include=np.number)
    corr = num_df.corr().round(2)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr.values, cmap='RdYlGn', vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(corr.columns, fontsize=9)
    for i in range(len(corr)):
        for j in range(len(corr.columns)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha='center', va='center', fontsize=7)
    ax.set_title('Feature Correlation Matrix', fontsize=13, fontweight='bold')
    report['corr_plot'] = _save(fig, 'correlation.png')

    # ── 6. Price distribution ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(df['Close'], bins=60, color='#0891B2', edgecolor='white', linewidth=0.4)
    ax.axvline(df['Close'].mean(), color='red', linestyle='--', label=f"Mean ₹{df['Close'].mean():.1f}")
    ax.set_title('Close Price Distribution', fontsize=13, fontweight='bold')
    ax.set_xlabel('Price (₹)')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(alpha=0.3)
    report['dist_plot'] = _save(fig, 'distribution.png')

    report['stats'] = {
        'total_records': len(df),
        'date_range': f"{df['Date'].min().date()} to {df['Date'].max().date()}",
        'close_min': round(df['Close'].min(), 2),
        'close_max': round(df['Close'].max(), 2),
        'close_mean': round(df['Close'].mean(), 2),
        'close_std': round(df['Close'].std(), 2),
    }
    return report


if __name__ == '__main__':
    r = run_eda()
    print("EDA complete. Plots saved:")
    for k, v in r.items():
        if 'plot' in k:
            print(' ', v)

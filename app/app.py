"""
app.py – Flask web application for the Stock Price Prediction System.
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import  sys, json, base64
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from flask import Flask, render_template, jsonify, request, send_from_directory
from preprocessing import load_data, preprocess
from eda import run_eda
from feature_engineering import engineer_features
from train import train, load_model
from evaluate import evaluate

app = Flask(__name__)

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.join(BASE_DIR, '..')
VISUALS    = os.path.join(ROOT_DIR, 'visuals')
DATA_RAW   = os.path.join(ROOT_DIR, 'data', 'raw', 'CIPLA.csv')
MODEL_PATH = os.path.join(ROOT_DIR, 'models', 'xgboost_model.pkl')
FEATURED_PATH = os.path.join(ROOT_DIR, 'data', 'processed', 'featured_data.csv')
PROCESSED_PATH = os.path.join(ROOT_DIR, 'data', 'processed', 'processed_data.csv')

# Server-side state (survives page refresh — data lives on server)
STATE = {}

def img64(path):
    try:
        with open(path, 'rb') as f:
            return base64.b64encode(f.read()).decode()
    except FileNotFoundError:
        return ''

def _load_clean_df():
    """Load df_clean from disk if not in memory."""
    import pandas as pd
    if 'df_clean' not in STATE:
        if os.path.exists(PROCESSED_PATH):
            STATE['df_clean'] = pd.read_csv(PROCESSED_PATH, parse_dates=['Date'])
    return STATE.get('df_clean')

def _load_feat_df():
    """Load df_feat from disk if not in memory."""
    import pandas as pd
    if 'df_feat' not in STATE:
        if os.path.exists(FEATURED_PATH):
            STATE['df_feat'] = pd.read_csv(FEATURED_PATH, parse_dates=['Date'])
    return STATE.get('df_feat')

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/visuals/<path:filename>')
def serve_visual(filename):
    return send_from_directory(VISUALS, filename)

# Step 1 – Preprocess
@app.route('/api/preprocess', methods=['POST'])
def api_preprocess():
    try:
        df_raw = load_data(DATA_RAW)
        df_clean, report = preprocess(df_raw)
        STATE['df_clean'] = df_clean
        report['missing_before'] = {k: int(v) for k, v in report['missing_before'].items()}
        report['missing_after']  = {k: int(v) for k, v in report['missing_after'].items()}
        for row in report.get('sample_rows', []):
            for k, v in row.items(): row[k] = str(v)
        return jsonify({'ok': True, 'report': report})
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'ok': False, 'error': str(e)}), 500

# Step 2 – EDA  (auto-loads clean df from disk if needed)
@app.route('/api/eda', methods=['POST'])
def api_eda():
    try:
        df = _load_clean_df()   # works even after page refresh
        report = run_eda(df)    # df=None falls back to processed_data.csv
        for key in ['close_history_plot', 'volume_plot', 'ohlc_plot', 'corr_plot', 'dist_plot']:
            if key in report:
                report[key] = img64(report[key])
        desc_df = {}
        for col, vals in report.get('describe', {}).items():
            desc_df[col] = {}
            for k, v in vals.items():
                try:    desc_df[col][k] = round(float(v), 2)
                except: desc_df[col][k] = str(v)
        report['describe'] = desc_df
        return jsonify({'ok': True, 'report': report})
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'ok': False, 'error': str(e)}), 500

# Step 3 – Features
@app.route('/api/features', methods=['POST'])
def api_features():
    try:
        df = _load_clean_df()
        df_feat, report = engineer_features(df)
        STATE['df_feat'] = df_feat
        clean_sample = []
        for row in report.get('sample', []):
            clean_row = {}
            for k, v in row.items():
                try:    clean_row[k] = round(float(v), 4)
                except: clean_row[k] = str(v)
            clean_sample.append(clean_row)
        report['sample'] = clean_sample
        return jsonify({'ok': True, 'report': report})
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'ok': False, 'error': str(e)}), 500

# Step 4 – Train
@app.route('/api/train', methods=['POST'])
def api_train():
    try:
        df_feat = _load_feat_df()
        model, scaler, X_test_s, y_test, dates_test, meta = train(df_feat)
        STATE['model']  = model
        STATE['scaler'] = scaler
        meta['feature_importance'] = [[f, round(float(v), 6)] for f, v in meta['feature_importance']]
        return jsonify({'ok': True, 'meta': meta})
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'ok': False, 'error': str(e)}), 500

# Step 5 – Evaluate
@app.route('/api/evaluate', methods=['POST'])
def api_evaluate():
    try:
        df_feat = _load_feat_df()
        model  = STATE.get('model')
        scaler = STATE.get('scaler')
        if model is None or scaler is None:
            model, scaler, _ = load_model()
        report = evaluate(model, scaler, df_feat)
        for key in ['pred_plot', 'residual_plot', 'importance_plot']:
            if key in report: report[key] = img64(report[key])
        return jsonify({'ok': True, 'report': report})
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'ok': False, 'error': str(e)}), 500

# Run all
@app.route('/api/run_all', methods=['POST'])
def api_run_all():
    try:
        df_raw = load_data(DATA_RAW)
        df_clean, pre = preprocess(df_raw)
        STATE['df_clean'] = df_clean
        _ = run_eda(df_clean)
        df_feat, _ = engineer_features(df_clean)
        STATE['df_feat'] = df_feat
        model, scaler, _, _, _, meta = train(df_feat)
        STATE['model']  = model
        STATE['scaler'] = scaler
        report = evaluate(model, scaler, df_feat)
        for key in ['pred_plot', 'residual_plot', 'importance_plot']:
            if key in report: report[key] = img64(report[key])
        return jsonify({'ok': True,
            'metrics': {k: report[k] for k in ['rmse','mae','r2','mape']},
            'report':  report})
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'ok': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)

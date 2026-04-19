"""
main.py
End-to-end pipeline runner (no Flask). Run once to pre-train the model.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from preprocessing import load_data, preprocess
from eda import run_eda
from feature_engineering import engineer_features
from train import train
from evaluate import evaluate


def main():
    print("=" * 60)
    print("  CIPLA Stock Price Prediction – Full Pipeline")
    print("=" * 60)

    # Step 1 – Preprocessing
    print("\n[1/5] Preprocessing …")
    df_raw = load_data()
    df_clean, pre_report = preprocess(df_raw)
    print(f"      Shape: {pre_report['shape_before']} → {pre_report['shape_after']}")

    # Step 2 – EDA
    print("\n[2/5] Exploratory Data Analysis …")
    eda_report = run_eda(df_clean)
    print(f"      Records: {eda_report['stats']['total_records']}, "
          f"Range: {eda_report['stats']['date_range']}")

    # Step 3 – Feature Engineering
    print("\n[3/5] Feature Engineering …")
    df_feat, feat_report = engineer_features(df_clean)
    print(f"      Features: {feat_report['total_features']}, "
          f"Shape: {feat_report['shape_after']}")

    # Step 4 – Training
    print("\n[4/5] Training XGBoost …")
    model, scaler, X_test_s, y_test, dates_test, train_meta = train(df_feat)
    print(f"      Train: {train_meta['train_size']} | Test: {train_meta['test_size']}")

    # Step 5 – Evaluation
    print("\n[5/5] Evaluating …")
    eval_report = evaluate(model, scaler, df_feat)
    print(f"      RMSE : ₹{eval_report['rmse']}")
    print(f"      MAE  : ₹{eval_report['mae']}")
    print(f"      R²   : {eval_report['r2']}")
    print(f"      MAPE : {eval_report['mape']}%")

    print("\n✅  Pipeline complete. Launch the web app with:")
    print("    cd app && python app.py")
    print("=" * 60)


if __name__ == '__main__':
    main()

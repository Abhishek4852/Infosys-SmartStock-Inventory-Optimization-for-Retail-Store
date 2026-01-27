import pandas as pd
import lightgbm as lgb
import pickle
import os
import json
import numpy as np

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

def train_lgbm(data_path, model_path, config_path):
    print("Loading data for LightGBM...")
    df = pd.read_csv(data_path)

    # Sort by time
    df = df.sort_values("Date")

    # Drop NaN from lag & rolling features
    df = df.dropna()

    features = [
        'Store', 'Dept', 'IsHoliday', 'Temperature', 'Fuel_Price',
        'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5',
        'CPI', 'Unemployment', 'Size',
        'Year', 'Month', 'Week', 'Day', 'DayOfWeek',
        'Lag_1', 'Lag_2', 'Lag_3', 'Lag_4', 'Lag_8', 'Lag_12',
        'Lag_16', 'Lag_20', 'Lag_24',
        'RollingMean_4', 'RollingStd_4',
        'RollingMean_12', 'RollingStd_12'
    ]

    X = df[features]
    y = df["Weekly_Sales"]

    # ==============================
    # TRAIN / VALIDATION SPLIT
    # ==============================

    split_index = int(len(df) * 0.8)

    X_train, X_val = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_val = y.iloc[:split_index], y.iloc[split_index:]

    print(f"Training rows: {len(X_train)}")
    print(f"Validation rows: {len(X_val)}")

    # ==============================
    # MODEL TRAINING
    # ==============================

    params = {
        "n_estimators": 250,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "verbose": -1
    }

    model = lgb.LGBMRegressor(**params)

    print("Training LightGBM model...")
    model.fit(X_train, y_train)

    # ==============================
    # EVALUATION
    # ==============================

    print("\n--- LightGBM Evaluation Metrics ---")

    y_pred = model.predict(X_val)

    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)

    mape = np.mean(np.abs((y_val - y_pred) / y_val)) * 100

    print(f"RMSE : {rmse:,.2f}")
    print(f"MAE  : {mae:,.2f}")
    print(f"MAPE : {mape:.2f}%")
    print(f"R²   : {r2:.4f}")

    # ==============================
    # SAVE MODEL
    # ==============================

    print(f"\nSaving LightGBM model to {model_path}...")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    # Save feature list for inference
    feature_list_path = os.path.join(
        os.path.dirname(model_path),
        "feature_list.json"
    )

    with open(feature_list_path, "w") as f:
        json.dump(features, f)

    print("LightGBM training complete ✅")
    print(rmse)

from pathlib import Path

# Detect project root (parent directory of 'src/training')
ROOT_DIR = Path(__file__).resolve().parent.parent.parent

if __name__ == "__main__":
    train_lgbm(
        str(ROOT_DIR / "data/processed/sales_features.csv"),
        str(ROOT_DIR / "model_artifacts/lgbm_model.pkl"),
        None
    )

import pandas as pd
import xgboost as xgb
import pickle
import os
import json
import numpy as np

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from pathlib import Path

# Detect project root
ROOT_DIR = Path(__file__).resolve().parent.parent.parent

def train_xgb(data_path, model_path):
    print("Loading data for XGBoost...")
    df = pd.read_csv(data_path)

    df = df.sort_values("Date")
    df = df.dropna()

    # ==============================
    # LOAD FEATURE LIST
    # ==============================

    feature_list_path = ROOT_DIR / "model_artifacts/feature_list.json"
    with open(feature_list_path, "r") as f:
        features = json.load(f)

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
    # MODEL PARAMETERS
    # ==============================

    params = {
        "n_estimators": 250,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "tree_method": "hist",
        "random_state": 42,
        "objective": "reg:squarederror",
        "verbosity": 0
    }

    model = xgb.XGBRegressor(**params)

    # ==============================
    # TRAIN MODEL
    # ==============================

    print("Training XGBoost model...")
    model.fit(X_train, y_train)

    # ==============================
    # EVALUATION
    # ==============================

    print("\n--- XGBoost Evaluation Metrics ---")

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

    print(f"\nSaving XGBoost model to {model_path}...")

    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    print("XGBoost training complete ✅")
    print(rmse)

if __name__ == "__main__":
    train_xgb(
        str(ROOT_DIR / "data/processed/sales_features.csv"),
        str(ROOT_DIR / "model_artifacts/xgb_model.pkl")
    )

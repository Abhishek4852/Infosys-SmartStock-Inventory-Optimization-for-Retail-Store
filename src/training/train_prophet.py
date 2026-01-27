import pandas as pd
from prophet import Prophet
import pickle
import os
import numpy as np

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

def train_prophet(data_path, model_path):
    print("Loading data for Prophet...")
    df = pd.read_csv(data_path)

    # ==============================
    # AGGREGATE SALES BY DATE
    # ==============================

    prophet_df = (
        df.groupby("Date")["Weekly_Sales"]
        .mean()
        .reset_index()
    )

    prophet_df.columns = ["ds", "y"]

    prophet_df = prophet_df.sort_values("ds")

    # ==============================
    # TRAIN / VALIDATION SPLIT
    # ==============================

    split_index = int(len(prophet_df) * 0.8)

    train_df = prophet_df.iloc[:split_index]
    val_df = prophet_df.iloc[split_index:]

    print(f"Training rows: {len(train_df)}")
    print(f"Validation rows: {len(val_df)}")

    # ==============================
    # TRAIN MODEL
    # ==============================

    print("Training Prophet model...")

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False
    )

    model.fit(train_df)

    # ==============================
    # EVALUATION
    # ==============================

    print("\n--- Prophet Evaluation Metrics ---")

    future = model.make_future_dataframe(
        periods=len(val_df),
        freq="W"
    )

    forecast = model.predict(future)

    # Match validation range
    y_true = val_df["y"].values
    y_pred = forecast.iloc[-len(val_df):]["yhat"].values

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    print(f"RMSE : {rmse:,.2f}")
    print(f"MAE  : {mae:,.2f}")
    print(f"MAPE : {mape:.2f}%")
    print(f"R²   : {r2:.4f}")

    # ==============================
    # SAVE MODEL
    # ==============================

    print(f"\nSaving Prophet model to {model_path}...")

    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    print("Prophet training complete ✅")
    print(rmse)

from pathlib import Path

# Detect project root
ROOT_DIR = Path(__file__).resolve().parent.parent.parent

if __name__ == "__main__":
    train_prophet(
        str(ROOT_DIR / "data/processed/sales_features.csv"),
        str(ROOT_DIR / "model_artifacts/prophet_model.pkl")
    )

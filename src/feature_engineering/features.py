import pandas as pd
import numpy as np
import os

def create_features(df_path, output_path):
    """
    Standard feature engineering process for Walmart sales data.
    """
    print("Loading cleaned data...")
    df = pd.read_csv(df_path)
    df['Date'] = pd.to_datetime(df['Date'])

    print("Sorting data by Store, Dept, and Date...")
    df = df.sort_values(['Store', 'Dept', 'Date'])

    print("Adding calendar features...")
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Week'] = df['Date'].dt.isocalendar().week.astype(int)
    df['Day'] = df['Date'].dt.day
    df['DayOfWeek'] = df['Date'].dt.weekday
    df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)
    df['IsMonthStart'] = df['Date'].dt.is_month_start.astype(int)
    df['IsMonthEnd'] = df['Date'].dt.is_month_end.astype(int)

    print("Adding lag features...")
    lags = [1, 2, 3, 4, 8, 12, 16, 20, 24]
    for lag in lags:
        df[f'Lag_{lag}'] = df.groupby(['Store', 'Dept'])['Weekly_Sales'].shift(lag)

    print("Adding rolling statistics...")
    windows = [4, 12]
    for window in windows:
        df[f'RollingMean_{window}'] = df.groupby(['Store', 'Dept'])['Weekly_Sales'].transform(
            lambda x: x.shift(1).rolling(window=window).mean()
        )
        df[f'RollingStd_{window}'] = df.groupby(['Store', 'Dept'])['Weekly_Sales'].transform(
            lambda x: x.shift(1).rolling(window=window).std()
        )

    print(f"Saving features to {output_path}...")
    df.to_csv(output_path, index=False)
    return df

from pathlib import Path

# Detect project root
ROOT_DIR = Path(__file__).resolve().parent.parent.parent

if __name__ == "__main__":
    create_features(
        str(ROOT_DIR / "data/processed/sales_cleaned.csv"),
        str(ROOT_DIR / "data/processed/sales_features.csv")
    )

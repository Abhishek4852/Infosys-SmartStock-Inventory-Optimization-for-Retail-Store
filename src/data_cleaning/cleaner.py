import pandas as pd
import numpy as np
import os

def clean_data(train_path, stores_path, features_path, output_path):
    """
    Standard data cleaning process for Walmart sales data.
    """
    print("Loading data...")
    train = pd.read_csv(train_path)
    stores = pd.read_csv(stores_path)
    features = pd.read_csv(features_path)

    print("Merging datasets...")
    # Merge train with stores and features
    df = train.merge(stores, on='Store', how='left')
    df = df.merge(features, on=['Store', 'Date', 'IsHoliday'], how='left')

    print("Handling missing values...")
    # Fill MarkDown missing values with 0
    markdown_cols = ['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']
    for col in markdown_cols:
        df[col] = df[col].fillna(0)

    # Convert Date to datetime
    df['Date'] = pd.to_datetime(df['Date'])

    print(f"Saving cleaned data to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    return df

from pathlib import Path

# Detect project root
ROOT_DIR = Path(__file__).resolve().parent.parent.parent

if __name__ == "__main__":
    clean_data(
        str(ROOT_DIR / "data/raw/train.csv"),
        str(ROOT_DIR / "data/raw/stores.csv"),
        str(ROOT_DIR / "data/raw/features.csv"),
        str(ROOT_DIR / "data/processed/sales_cleaned.csv")
    )

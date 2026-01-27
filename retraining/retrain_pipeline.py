import subprocess
import os
from pathlib import Path
import sys
import json

ROOT_DIR = Path(__file__).resolve().parent.parent

def calculate_weights(rmse_dict):
    """
    weight = (1 / rmse) / sum(1 / rmse)
    """
    inv = {k: 1/v for k, v in rmse_dict.items() if v > 0}
    total = sum(inv.values())

    weights = {k: inv[k] / total for k in inv}
    return weights

def get_rmse_from_output(output):
    """
    Extracts the last line of the output and converts it to float.
    """
    lines = output.strip().splitlines()
    if not lines:
        return 0.0
    try:
        return float(lines[-1])
    except ValueError:
        print(f"Error parsing RMSE from output: {lines[-1]}")
        return 0.0

def run_pipeline():
    print("\n--- STARTING FULL TRAINING PIPELINE ---\n")

    python_exe = sys.executable
    rmse_scores = {}

    # 1. Data Cleaning
    print("[1/5] Running Data Cleaning...")
    subprocess.run(
        [python_exe, str(ROOT_DIR / "src/data_cleaning/cleaner.py")],
        check=True,
        cwd=str(ROOT_DIR)
    )

    # 2. Feature Engineering
    print("\n[2/5] Running Feature Engineering...")
    subprocess.run(
        [python_exe, str(ROOT_DIR / "src/feature_engineering/features.py")],
        check=True,
        cwd=str(ROOT_DIR)
    )

    # 3. Train Models & Capture RMSE
    print("\n[3/5] Training Models...")

    print("Training LightGBM...")
    output_lgbm = subprocess.check_output(
        [python_exe, str(ROOT_DIR / "src/training/train_lgbm.py")],
        cwd=str(ROOT_DIR)
    ).decode()
    rmse_scores["lgbm"] = get_rmse_from_output(output_lgbm)

    print("Training XGBoost...")
    output_xgb = subprocess.check_output(
        [python_exe, str(ROOT_DIR / "src/training/train_xgb.py")],
        cwd=str(ROOT_DIR)
    ).decode()
    rmse_scores["xgb"] = get_rmse_from_output(output_xgb)

    print("Training Prophet...")
    output_prophet = subprocess.check_output(
        [python_exe, str(ROOT_DIR / "src/training/train_prophet.py")],
        cwd=str(ROOT_DIR)
    ).decode()
    rmse_scores["prophet"] = get_rmse_from_output(output_prophet)

    # 4. Calculate weights
    print("\n[4/5] Calculating Ensemble Weights...")
    weights = calculate_weights(rmse_scores)

    ensemble_config = {
        "rmse": rmse_scores,
        "weights": weights
    }

    config_path = ROOT_DIR / "model_artifacts" / "ensemble_config.json"
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(ensemble_config, f, indent=4)

    print("✅ Ensemble Weights Updated")
    print(json.dumps(ensemble_config, indent=4))

    # 5. Model size check
    print("\n[5/5] Checking model sizes (< 80MB):")

    for art in ["lgbm_model.pkl", "xgb_model.pkl", "prophet_model.pkl"]:
        path = ROOT_DIR / "model_artifacts" / art
        if path.exists():
            size = os.path.getsize(path) / (1024 * 1024)
            print(f" - {art}: {size:.2f} MB")
        else:
            print(f" - {art}: NOT FOUND")

    print("\n--- PIPELINE COMPLETE ---\n")

if __name__ == "__main__":
    run_pipeline()

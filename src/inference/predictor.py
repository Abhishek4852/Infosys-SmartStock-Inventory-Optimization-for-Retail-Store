import pickle
import pandas as pd
import numpy as np
import json
import os

class SalesPredictor:
    def __init__(self, model_dir='model_artifacts'):
        self.model_dir = model_dir
        self.lgbm_model = self._load_model('lgbm_model.pkl')
        self.xgb_model = self._load_model('xgb_model.pkl')
        self.prophet_model = self._load_model('prophet_model.pkl')
        
        with open(os.path.join(model_dir, 'feature_list.json'), 'r') as f:
            self.features = json.load(f)
            
        with open(os.path.join(model_dir, 'ensemble_config.json'), 'r') as f:
            self.config = json.load(f)
            
    def _load_model(self, model_name):
        path = os.path.join(self.model_dir, model_name)
        with open(path, 'rb') as f:
            return pickle.load(f)
            
    def predict(self, input_data):
        # input_data is a dict or dataframe with necessary features
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        else:
            df = input_data
            
        # LGBM & XGB Predictions
        lgbm_pred = self.lgbm_model.predict(df[self.features])[0]
        xgb_pred = self.xgb_model.predict(df[self.features])[0]
        
        # Prophet Prediction (Simplified - using day/month trend)
        # Note: In a real scenario, we'd pass the date to the prophet model
        # Here we just use the weights for the ensemble
        # For simplicity, we'll assume the prophet model returns a baseline or we skip it for single row if not applicable
        # But per requirements, all must be used.
        
        # We need a date for Prophet
        ds = pd.to_datetime(f"{df['Year'].iloc[0]}-{df['Month'].iloc[0]}-{df['Day'].iloc[0]}")
        prophet_df = pd.DataFrame({'ds': [ds]})
        prophet_forecast = self.prophet_model.predict(prophet_df)
        prophet_pred = prophet_forecast['yhat'].iloc[0]
        
        # Ensemble weights
        w_lgbm = self.config['weights']['lgbm']
        w_xgb = self.config['weights']['xgb']
        w_prophet = self.config['weights']['prophet']
        
        final_pred = (w_lgbm * lgbm_pred) + (w_xgb * xgb_pred) + (w_prophet * prophet_pred)
        
        return {
            'next_week_sales': round(final_pred, 2),
            'next_month_sales': round(final_pred * 4, 2), # Simplified scaling
            'next_3_month_sales': round(final_pred * 12, 2) # Simplified scaling
        }

if __name__ == "__main__":
    # Create default config if not exists
    # Detect project root
    ROOT_DIR = Path(__file__).resolve().parent.parent.parent
    config_path = str(ROOT_DIR / "model_artifacts/ensemble_config.json")
    if not os.path.exists(config_path):
        config = {
            "weights": {
                "lgbm": 0.4,
                "xgb": 0.3,
                "prophet": 0.3
            }
        }
        with open(config_path, 'w') as f:
            json.dump(config, f)

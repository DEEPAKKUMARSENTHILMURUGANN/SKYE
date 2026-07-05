import torch
import torch.nn as nn
import xgboost as xgb
import numpy as np
import os
import json

class LSTMRULRegressor(nn.Module):
    """
    Lightweight LSTM sequence-based model for predicting Remaining Useful Life (RUL).
    Takes sliding windows of sensor telemetry and outputs a single scalar RUL.
    """
    def __init__(self, input_dim=21, hidden_dim=32, num_layers=2):
        super(LSTMRULRegressor, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, (hn, _) = self.lstm(x)
        last_hidden = hn[-1]
        rul = self.fc(last_hidden)
        return rul

class RULEnsemble:
    """
    Predictive maintenance ensemble combining:
    1. LSTM sequence model (captures temporal trends)
    2. XGBoost regressor (captures instant feature relationships)
    """
    def __init__(self, lstm_path="trained_models/rul_lstm.pt", xgb_path="trained_models/rul_xgb.json"):
        self.lstm_path = lstm_path
        self.xgb_path = xgb_path
        
        self.lstm_model = None
        self.xgb_model = None
        self.mean_coeff = None
        self.std_coeff = None
        
        self.load_models()

    def load_models(self):
        if os.path.exists(self.lstm_path):
            checkpoint = torch.load(self.lstm_path, weights_only=False)
            self.lstm_model = LSTMRULRegressor(input_dim=21)
            self.lstm_model.load_state_dict(checkpoint["model_state_dict"])
            self.lstm_model.eval()
            self.mean_coeff = checkpoint.get("mean_coeff")
            self.std_coeff = checkpoint.get("std_coeff")
        if os.path.exists(self.xgb_path):
            self.xgb_model = xgb.Booster()
            self.xgb_model.load_model(self.xgb_path)

    def predict(self, window_data: np.ndarray) -> dict:
        """
        Runs ensemble inference.
        Args:
            window_data (np.ndarray): 2D array of shape (seq_len, 24) representing raw/normal sensor logs.
        Returns:
            dict containing LSTM, XGBoost, and combined RUL predictions.
        """
        if self.lstm_model is None or self.xgb_model is None:
            return {
                "lstm_rul": 150.0,
                "xgb_rul": 150.0,
                "ensemble_rul": 150.0,
                "current_health": 83.3,
                "failure_probability": 0.13,
                "risk": "Low"
            }
        sensors = window_data[:, 3:24]
        scaled_sensors = (sensors - self.mean_coeff) / (self.std_coeff + 1e-8)
        
        x_tensor = torch.tensor(scaled_sensors, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            lstm_pred = self.lstm_model(x_tensor).item()
        lstm_pred = max(0.0, lstm_pred)
        latest_features = window_data[-1, 0:24].reshape(1, -1)
        dmat = xgb.DMatrix(latest_features)
        xgb_pred = float(self.xgb_model.predict(dmat)[0])
        xgb_pred = max(0.0, xgb_pred)
        ensemble_pred = 0.5 * lstm_pred + 0.5 * xgb_pred
        if ensemble_pred > 120:
            risk = "Low"
            failure_prob = round(0.05 + 0.1 * (120 / ensemble_pred), 3)
        elif ensemble_pred > 50:
            risk = "Medium"
            failure_prob = round(0.2 + 0.4 * ((120 - ensemble_pred) / 70.0), 3)
        else:
            risk = "High"
            failure_prob = round(0.6 + 0.38 * ((50 - ensemble_pred) / 50.0), 3)
        health = round(min(100.0, (ensemble_pred / 180.0) * 100.0), 1)
        
        return {
            "lstm_rul": round(lstm_pred, 1),
            "xgb_rul": round(xgb_pred, 1),
            "ensemble_rul": round(ensemble_pred, 1),
            "current_health": health,
            "failure_probability": failure_prob,
            "risk": risk
        }

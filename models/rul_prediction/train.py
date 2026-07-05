import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import xgboost as xgb
from models.rul_prediction.model import LSTMRULRegressor

def prepare_xgb_data(df):
    """Prepares flat features and targets for XGBoost regressor."""
    rul_list = []
    for uid, group in df.groupby("unit_number"):
        max_cycle = group["time_in_cycles"].max()
        engine_ruls = max_cycle - group["time_in_cycles"].values
        engine_ruls = np.clip(engine_ruls, 0, 130)
        rul_list.extend(engine_ruls)
        
    df["RUL"] = rul_list
    feat_cols = ["op_setting_1", "op_setting_2", "op_setting_3"] + [f"s{i}" for i in range(1, 22)]
    X = df[feat_cols].values
    y = df["RUL"].values
    return X, y

def prepare_lstm_data(df, window_size=30):
    """Prepares sliding windows of sensor telemetry and targets for LSTM."""
    sensor_cols = [f"s{i}" for i in range(1, 22)]
    sensors = df[sensor_cols].values
    mean_coeff = np.mean(sensors, axis=0)
    std_coeff = np.std(sensors, axis=0)
    scaled_sensors = (sensors - mean_coeff) / (std_coeff + 1e-8)
    df_scaled = df.copy()
    df_scaled[sensor_cols] = scaled_sensors
    
    rul_list = []
    for uid, group in df_scaled.groupby("unit_number"):
        max_cycle = group["time_in_cycles"].max()
        engine_ruls = max_cycle - group["time_in_cycles"].values
        engine_ruls = np.clip(engine_ruls, 0, 130)
        rul_list.extend(engine_ruls)
        
    df_scaled["RUL"] = rul_list
    
    X_list = []
    y_list = []
    for uid, group in df_scaled.groupby("unit_number"):
        sens_vals = group[sensor_cols].values
        ruls = group["RUL"].values
        
        for idx in range(len(group)):
            if idx < window_size - 1:
                pad_len = (window_size - 1) - idx
                pad = np.repeat(sens_vals[0:1], pad_len, axis=0)
                window = np.vstack([pad, sens_vals[0 : idx + 1]])
            else:
                window = sens_vals[idx - window_size + 1 : idx + 1]
                
            X_list.append(window)
            y_list.append(ruls[idx])
            
    X = np.array(X_list)
    y = np.array(y_list)
    
    return X, y, mean_coeff, std_coeff

def train_lstm(X_train, y_train, epochs=25, batch_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMRULRegressor(input_dim=21).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.003)
    
    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    
    dataset_size = len(X_tensor)
    model.train()
    
    print("\nTraining RUL LSTM Model...")
    for epoch in range(epochs):
        permutation = torch.randperm(dataset_size)
        epoch_loss = 0.0
        
        for i in range(0, dataset_size, batch_size):
            indices = permutation[i:i+batch_size]
            batch_x = X_tensor[indices].to(device)
            batch_y = y_tensor[indices].to(device)
            
            optimizer.zero_grad()
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * len(batch_x)
            
        epoch_loss /= dataset_size
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:02d}/{epochs:02d} | Loss: {epoch_loss:.2f}")
            
    return model

def main():
    train_file = "datasets/train_FD001.csv"
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"Training dataset not found: {train_file}. Run generate_cmapss.py first.")
        
    df = pd.read_csv(train_file)
    print("Preparing data for XGBoost...")
    X_xgb, y_xgb = prepare_xgb_data(df)
    
    print("Training XGBoost Regressor...")
    dtrain = xgb.DMatrix(X_xgb, label=y_xgb)
    params = {
        "max_depth": 5,
        "eta": 0.1,
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "seed": 42
    }
    xgb_booster = xgb.train(params, dtrain, num_boost_round=60)
    print("\nPreparing data for LSTM...")
    X_lstm, y_lstm, mean_coeff, std_coeff = prepare_lstm_data(df, window_size=30)
    lstm_model = train_lstm(X_lstm, y_lstm, epochs=20, batch_size=64)
    os.makedirs("trained_models", exist_ok=True)
    xgb_save_path = "trained_models/rul_xgb.json"
    xgb_booster.save_model(xgb_save_path)
    print(f"XGBoost model successfully saved to: {xgb_save_path}")
    lstm_save_path = "trained_models/rul_lstm.pt"
    checkpoint = {
        "model_state_dict": lstm_model.state_dict(),
        "mean_coeff": mean_coeff,
        "std_coeff": std_coeff
    }
    torch.save(checkpoint, lstm_save_path)
    print(f"LSTM model successfully saved to: {lstm_save_path}")

if __name__ == "__main__":
    main()

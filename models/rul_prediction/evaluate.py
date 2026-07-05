import os
import json
import pandas as pd
import numpy as np
from models.rul_prediction.model import RULEnsemble

def load_test_ground_truth():
    """Loads testing telemetry and true RUL values."""
    test_file = "datasets/test_FD001.csv"
    rul_file = "datasets/RUL_FD001.txt"
    
    if not os.path.exists(test_file) or not os.path.exists(rul_file):
        raise FileNotFoundError("Test data files not found. Run generate_cmapss.py and train.py first.")
        
    df_test = pd.read_csv(test_file)
    with open(rul_file, "r") as f:
        true_ruls = [int(line.strip()) for line in f.readlines()]
        
    return df_test, true_ruls

def main():
    print("=" * 70)
    print("Evaluating SKYE Remaining Useful Life (RUL) Prediction Ensemble")
    print("=" * 70)
    df_test, true_ruls = load_test_ground_truth()
    ensemble = RULEnsemble()
    feat_cols = ["op_setting_1", "op_setting_2", "op_setting_3"] + [f"s{i}" for i in range(1, 22)]
    
    predictions = []
    for uid, group in df_test.groupby("unit_number"):
        window_size = 30
        group_len = len(group)
        group_feats = group[feat_cols].values
        
        if group_len < window_size:
            pad_len = window_size - group_len
            pad = np.repeat(group_feats[0:1], pad_len, axis=0)
            window = np.vstack([pad, group_feats])
        else:
            window = group_feats[-window_size:]
            
        pred_res = ensemble.predict(window)
        predictions.append(pred_res)
    pred_ruls = [p["ensemble_rul"] for p in predictions]
    true_ruls = true_ruls[:len(pred_ruls)]
    
    rmse = np.sqrt(np.mean((np.array(pred_ruls) - np.array(true_ruls)) ** 2))
    mae = np.mean(np.abs(np.array(pred_ruls) - np.array(true_ruls)))
    print(f"\nSubsystem: Engine Turbofan Bearings")
    print("-" * 70)
    print(f"{'Engine ID':<10} | {'True RUL (Hrs)':<15} | {'Pred RUL (Hrs)':<15} | {'Error (Hrs)':<12} | {'Risk':<8}")
    print("-" * 70)
    for idx, (true_val, pred_val) in enumerate(zip(true_ruls, pred_ruls)):
        err = pred_val - true_val
        risk = predictions[idx]["risk"]
        print(f"Engine {idx+1:02d}  | {true_val:<15d} | {pred_val:<15.1f} | {err:<+12.1f} | {risk:<8}")
    print("-" * 70)
    
    print(f"Accuracy Metrics:")
    print(f"  * Root Mean Squared Error (RMSE): {rmse:.2f} flight hours")
    print(f"  * Mean Absolute Error (MAE):     {mae:.2f} flight hours")
    print("-" * 70)
    os.makedirs("trained_models", exist_ok=True)
    report = {
        "rmse": float(rmse),
        "mae": float(mae),
        "test_count": len(pred_ruls),
        "predictions": pred_ruls,
        "true_values": true_ruls
    }
    with open("trained_models/rul_report.json", "w") as f:
        json.dump(report, f, indent=4)
    print("\nRUL Prediction Graph (Comparison Plot):")
    print(" (Legend: T = True RUL, P = Predicted RUL, scaled 0 to 140)")
    print("-" * 70)
    for idx, (t, p) in enumerate(zip(true_ruls, pred_ruls)):
        t_pos = int(min(140.0, t) / 140.0 * 50)
        p_pos = int(min(140.0, p) / 140.0 * 50)
        
        line = [" "] * 52
        line[t_pos] = "T"
        if line[p_pos] == "T":
            line[p_pos] = "X"          
        else:
            line[p_pos] = "P"
            
        graph_str = "".join(line)
        print(f"Eng {idx+1:02d} |{graph_str}|")
    print("-" * 70)
    
if __name__ == "__main__":
    main()

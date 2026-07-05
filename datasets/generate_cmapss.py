import os
import pandas as pd
import numpy as np

def generate_engine_trajectory(unit_id, max_cycles, is_train=True):
    """
    Generates a single run-to-failure or truncated engine trajectory.
    Replicates the format and degradation signatures of CMAPSS FD001.
    """
    np.random.seed(unit_id + (42 if is_train else 100))
    if is_train:
        cycles = max_cycles
    else:
        cycles = int(max_cycles * np.random.uniform(0.5, 0.85))
        
    records = []
    s_base = {
        "s1": 518.67, "s2": 642.3, "s3": 1589.7, "s4": 1400.6,
        "s5": 14.62, "s6": 21.61, "s7": 554.3, "s8": 2388.06,
        "s9": 9044.03, "s10": 1.3, "s11": 47.47, "s12": 521.66,
        "s13": 2388.02, "s14": 8138.62, "s15": 8.41, "s16": 0.03,
        "s17": 392.0, "s18": 2388.0, "s19": 100.0, "s20": 38.85, "s21": 23.31
    }
    
    for c in range(1, cycles + 1):
        deg_start = int(max_cycles * 0.5)
        if c > deg_start:
            t_ratio = (c - deg_start) / (max_cycles - deg_start)
            deg_factor = t_ratio ** 2
        else:
            deg_factor = 0.0
            
        record = {
            "unit_number": unit_id,
            "time_in_cycles": c,
            "op_setting_1": float(np.random.normal(0.001, 0.0015)),
            "op_setting_2": float(np.random.normal(0.0002, 0.0001)),
            "op_setting_3": 100.0
        }
        for s_idx in range(1, 22):
            s_name = f"s{s_idx}"
            val = s_base[s_name]
            noise = np.random.normal(0, val * 0.0015)
            
            if s_name == "s2":
                val += deg_factor * 15.0
            elif s_name == "s3":
                val += deg_factor * 25.0
            elif s_name == "s4":
                val += deg_factor * 20.0
            elif s_name == "s7":
                val -= deg_factor * 6.0
            elif s_name == "s11":
                val += deg_factor * 1.5
            elif s_name == "s12":
                val -= deg_factor * 4.5
            elif s_name == "s15":
                val += deg_factor * 0.12
            elif s_name == "s17":
                val += deg_factor * 5.0
            elif s_name == "s20":
                val -= deg_factor * 1.8
            elif s_name == "s21":
                val -= deg_factor * 1.2
                
            record[s_name] = round(float(val + noise), 4)
            
        records.append(record)
        
    return records, (max_cycles - cycles)

def generate_all_datasets():
    os.makedirs("datasets", exist_ok=True)
    
    train_records = []
    test_records = []
    test_ruls = []
    print("Generating synthetic CMAPSS train set...")
    for uid in range(1, 31):
        max_life = int(np.random.randint(130, 240))
        traj, _ = generate_engine_trajectory(uid, max_life, is_train=True)
        train_records.extend(traj)
    print("Generating synthetic CMAPSS test set and ground-truth RULs...")
    for uid in range(1, 16):
        max_life = int(np.random.randint(130, 240))
        traj, test_rul = generate_engine_trajectory(uid, max_life, is_train=False)
        test_records.extend(traj)
        test_ruls.append(test_rul)
        
    df_train = pd.DataFrame(train_records)
    df_test = pd.DataFrame(test_records)
    df_train.to_csv("datasets/train_FD001.csv", index=False)
    df_test.to_csv("datasets/test_FD001.csv", index=False)
    
    with open("datasets/RUL_FD001.txt", "w") as f:
        for rul in test_ruls:
            f.write(f"{rul}\n")
            
    print(f"Created datasets/train_FD001.csv with shape {df_train.shape}")
    print(f"Created datasets/test_FD001.csv with shape {df_test.shape}")
    print(f"Created datasets/RUL_FD001.txt with {len(test_ruls)} records.")

if __name__ == "__main__":
    generate_all_datasets()

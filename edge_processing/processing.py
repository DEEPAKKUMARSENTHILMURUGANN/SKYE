import numpy as np
from collections import deque
import time

class EdgeAIProcessor:
    """
    Onboard edge processing pipeline for real-time aircraft sensor data.
    Implements noise filtering, normalization, feature extraction, 
    and time-series windowing with minimal memory and compute overhead.
    """
    def __init__(self, window_size=30, ema_alpha=0.25):
        self.window_size = window_size
        self.ema_alpha = ema_alpha
        self.raw_window = deque(maxlen=window_size)
        self.filtered_window = deque(maxlen=window_size)
        self.normalized_window = deque(maxlen=window_size)
        self.last_filtered = {}
        self.bounds = {
            "altitude": (0.0, 31000.0),
            "speed": (0.0, 500.0),
            "engine_vibration": (0.0, 10.0),
            "egt": (300.0, 950.0),
            "oil_pressure": (20.0, 80.0),
            "n1_speed": (0.0, 110.0),
            "n2_speed": (0.0, 110.0),
            "fuel_flow": (0.0, 4.0),
            "hydraulic_pressure": (1500.0, 3300.0),
            "pump_efficiency": (50.0, 100.0),
            "actuator_response": (0.05, 1.0),
            "airframe_stress": (0.0, 500.0),
            "wing_load": (0.5, 2.5),
            "accel_x": (-1.0, 1.0),
            "accel_y": (-1.0, 1.0),
            "accel_z": (0.0, 2.5),
            "cabin_pressure": (8.0, 15.0),
            "bleed_temp": (80.0, 300.0),
            "ac_efficiency": (50.0, 100.0),
            "egt_vibe_ratio": (50.0, 1000.0),
            "pressure_ratio": (15.0, 70.0),
            "fuel_efficiency": (0.0, 2000.0),
            "stress_g_ratio": (10.0, 600.0)
        }
        self.feature_keys = list(self.bounds.keys())

    def filter_noise(self, raw_features: dict) -> dict:
        """Applies Exponential Moving Average (EMA) to smooth out raw sensor noise."""
        filtered = {}
        for key, val in raw_features.items():
            if key not in self.last_filtered:
                filtered[key] = val
            else:
                filtered[key] = self.ema_alpha * val + (1.0 - self.ema_alpha) * self.last_filtered[key]
            self.last_filtered[key] = filtered[key]
        return filtered

    def extract_features(self, filtered_features: dict) -> dict:
        """Extracts engineered virtual features from physical metrics."""
        feats = filtered_features.copy()
        feats["egt_vibe_ratio"] = feats["egt"] / (feats["engine_vibration"] + 1e-5)
        feats["pressure_ratio"] = feats["hydraulic_pressure"] / (feats["pump_efficiency"] + 1e-5)
        feats["fuel_efficiency"] = feats["speed"] / (feats["fuel_flow"] + 1e-5)
        feats["stress_g_ratio"] = feats["airframe_stress"] / (feats["wing_load"] + 1e-5)
        
        return feats

    def normalize(self, features: dict) -> dict:
        """MinMax normalizes all features to [0, 1] range based on bounds."""
        norm_feats = {}
        for key, val in features.items():
            if key in self.bounds:
                min_val, max_val = self.bounds[key]
                norm_feats[key] = (val - min_val) / (max_val - min_val + 1e-8)
                norm_feats[key] = max(0.0, min(1.0, norm_feats[key]))
            else:
                norm_feats[key] = val
        return norm_feats

    def process_telemetry(self, telemetry: dict) -> dict:
        """
        Main entry point for edge processing.
        Flattens raw data, applies filtering, normalization, and window updates.
        Returns the processed telemetry packet.
        """
        raw_flat = {
            "altitude": telemetry["altitude"],
            "speed": telemetry["speed"],
            **telemetry["engine"],
            **telemetry["hydraulic"],
            **telemetry["structural"],
            **telemetry["environmental"]
        }
        if "fatigue_cycles" in raw_flat:
            del raw_flat["fatigue_cycles"]
            
        self.raw_window.append(raw_flat)
        filtered_flat = self.filter_noise(raw_flat)
        self.filtered_window.append(filtered_flat)
        engineered_flat = self.extract_features(filtered_flat)
        normalized_flat = self.normalize(engineered_flat)
        self.normalized_window.append(normalized_flat)
        health_scores = self.calculate_heuristic_health(engineered_flat)
        
        return {
            "raw_flat": raw_flat,
            "filtered_flat": filtered_flat,
            "engineered_flat": engineered_flat,
            "normalized_flat": normalized_flat,
            "heuristic_health": health_scores,
            "mode": telemetry["mode"]
        }

    def get_sliding_window_array(self) -> np.ndarray:
        """
        Returns a normalized 2D numpy array of shape (window_size, num_features)
        suitable for feeding directly into LSTM or Transformer models.
        """
        if len(self.normalized_window) < self.window_size:
            pad_count = self.window_size - len(self.normalized_window)
            oldest = self.normalized_window[0] if self.normalized_window else {k: 0.0 for k in self.feature_keys}
            window_list = [oldest] * pad_count + list(self.normalized_window)
        else:
            window_list = list(self.normalized_window)
        arr = np.zeros((self.window_size, len(self.feature_keys)))
        for i, data in enumerate(window_list):
            for j, key in enumerate(self.feature_keys):
                arr[i, j] = data.get(key, 0.0)
        return arr

    def calculate_heuristic_health(self, feats: dict) -> dict:
        """Computes simple rule-based subsystem health indicators (0-100%)."""
        eng_vibe = feats["engine_vibration"]
        eng_egt = feats["egt"]
        eng_health = 100.0
        if eng_vibe > 3.0:
            eng_health -= min(50.0, (eng_vibe - 3.0) * 15.0)
        if eng_egt > 720.0:
            eng_health -= min(40.0, (eng_egt - 720.0) * 0.4)
        hyd_press = feats["hydraulic_pressure"]
        hyd_eff = feats["pump_efficiency"]
        hyd_health = 100.0
        if hyd_press < 2800.0:
            hyd_health -= min(60.0, (2800.0 - hyd_press) * 0.1)
        if hyd_eff < 90.0:
            hyd_health -= min(40.0, (90.0 - hyd_eff) * 2.0)
        stress = feats["airframe_stress"]
        load = feats["wing_load"]
        struc_health = 100.0
        if stress > 250.0:
            struc_health -= min(50.0, (stress - 250.0) * 0.3)
        if load > 1.3:
            struc_health -= min(30.0, (load - 1.3) * 50.0)
        press = feats["cabin_pressure"]
        bleed = feats["bleed_temp"]
        env_health = 100.0
        if press < 10.9:
            env_health -= min(60.0, (10.9 - press) * 30.0)
        if bleed > 215.0:
            env_health -= min(40.0, (bleed - 215.0) * 0.8)
        overall = (eng_health + hyd_health + struc_health + env_health) / 4.0
        
        return {
            "engine": round(max(0.0, eng_health), 1),
            "hydraulic": round(max(0.0, hyd_health), 1),
            "structural": round(max(0.0, struc_health), 1),
            "environmental": round(max(0.0, env_health), 1),
            "overall": round(max(0.0, overall), 1)
        }

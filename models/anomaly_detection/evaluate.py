import os
import torch
import numpy as np
from sensor_simulator.simulator import AircraftSensorSimulator
from edge_processing.processing import EdgeAIProcessor
from models.anomaly_detection.model import LSTMAutoencoder

class AnomalyEvaluator:
    """
    Edge inference class for loading the trained LSTM Autoencoder
    and evaluating telemetry sequences in real-time.
    """
    def __init__(self, model_path="trained_models/anomaly_detector.pt"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}. Please train the model first.")
            
        checkpoint = torch.load(model_path, weights_only=False)
        self.seq_len = checkpoint.get("seq_len", 30)
        self.input_dim = checkpoint.get("input_dim", 23)
        self.threshold = checkpoint.get("threshold", 0.01)
        
        self.model = LSTMAutoencoder(seq_len=self.seq_len, input_dim=self.input_dim)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def detect_anomaly(self, window_arr: np.ndarray):
        """
        Infers anomaly status, score, and subsystem health from a normalized sliding window.
        Returns:
            anomaly_score (float): normalized [0, 1] anomaly metric
            health_score (float): system health percentage [0, 100]
            status (str): "NORMAL", "DEGRADED", or "CRITICAL"
        """
        x = torch.tensor(window_arr, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            reconstructed = self.model(x)
            loss = torch.mean((reconstructed - x) ** 2).item()
        scaled_score = loss / (self.threshold * 1.8)
        scaled_score = min(1.0, max(0.0, scaled_score))
        health_score = 100.0 * (1.0 - scaled_score)
        
        if scaled_score < 0.35:
            status = "NORMAL"
        elif scaled_score < 0.60:
            status = "DEGRADED"
        else:
            status = "CRITICAL"
            
        return {
            "reconstruction_loss": loss,
            "anomaly_score": round(scaled_score, 4),
            "health_score": round(health_score, 1),
            "status": status
        }

def run_evaluation_demo():
    print("=" * 70)
    print("Evaluating SKYE Anomaly Detection Model")
    print("=" * 70)
    
    evaluator = AnomalyEvaluator()
    sim = AircraftSensorSimulator()
    processor = EdgeAIProcessor(window_size=30)
    
    print(f"Model threshold: {evaluator.threshold:.6f}")
    print("\n--- Testing NORMAL flight profile (5 steps) ---")
    for _ in range(5):
        telemetry = sim.get_next_telemetry()
        processed = processor.process_telemetry(telemetry)
        window = processor.get_sliding_window_array()
        result = evaluator.detect_anomaly(window)
        print(f"Step: {telemetry['step']:03d} | Loss: {result['reconstruction_loss']:.6f} | Anomaly Score: {result['anomaly_score']:.4f} | Health: {result['health_score']}% | Status: {result['status']}")
    print("\n--- Injecting BEARING WEAR (Gradual Degradation, 35 steps) ---")
    sim.set_mode("DEGRADATION")
    sim.inject_fault("bearing_wear")
    for i in range(35):
        telemetry = sim.get_next_telemetry()
        processed = processor.process_telemetry(telemetry)
        window = processor.get_sliding_window_array()
        result = evaluator.detect_anomaly(window)
        if i % 10 == 0 or i == 34:
            print(f"Step: {telemetry['step']:03d} | Loss: {result['reconstruction_loss']:.6f} | Anomaly Score: {result['anomaly_score']:.4f} | Health: {result['health_score']}% | Status: {result['status']}")
    print("\n--- Injecting SUDDEN HYDRAULIC LEAK (Fault Injection, 25 steps) ---")
    sim.inject_fault("hydraulic_leak")
    for i in range(25):
        telemetry = sim.get_next_telemetry()
        processed = processor.process_telemetry(telemetry)
        window = processor.get_sliding_window_array()
        result = evaluator.detect_anomaly(window)
        if i % 8 == 0 or i == 24:
            print(f"Step: {telemetry['step']:03d} | Loss: {result['reconstruction_loss']:.6f} | Anomaly Score: {result['anomaly_score']:.4f} | Health: {result['health_score']}% | Status: {result['status']}")
    print("\n--- Injecting COMPRESSOR SURGE (Sudden Anomaly, 25 steps) ---")
    sim.set_mode("SUDDEN_ANOMALY")
    sim.inject_fault("compressor_surge")
    for i in range(25):
        telemetry = sim.get_next_telemetry()
        processed = processor.process_telemetry(telemetry)
        window = processor.get_sliding_window_array()
        result = evaluator.detect_anomaly(window)
        if i % 8 == 0 or i == 24:
            print(f"Step: {telemetry['step']:03d} | Loss: {result['reconstruction_loss']:.6f} | Anomaly Score: {result['anomaly_score']:.4f} | Health: {result['health_score']}% | Status: {result['status']}")

if __name__ == "__main__":
    run_evaluation_demo()

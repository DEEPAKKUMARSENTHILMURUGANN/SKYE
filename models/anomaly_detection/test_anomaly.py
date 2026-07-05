import unittest
import torch
import numpy as np
import os
from models.anomaly_detection.model import LSTMAutoencoder
from models.anomaly_detection.evaluate import AnomalyEvaluator

class TestAnomalyDetection(unittest.TestCase):
    def test_model_forward_shape(self):
        batch_size = 4
        seq_len = 30
        input_dim = 23
        model = LSTMAutoencoder(seq_len=seq_len, input_dim=input_dim)
        x = torch.randn(batch_size, seq_len, input_dim)
        y = model(x)
        self.assertEqual(y.shape, x.shape)

    def test_evaluator_inference(self):
        model_path = "trained_models/anomaly_detector.pt"
        if not os.path.exists(model_path):
            self.skipTest("No trained model found. Run train.py first.")
            
        evaluator = AnomalyEvaluator(model_path=model_path)
        dummy_window = np.random.rand(30, 23)
        
        result = evaluator.detect_anomaly(dummy_window)
        
        self.assertIn("reconstruction_loss", result)
        self.assertIn("anomaly_score", result)
        self.assertIn("health_score", result)
        self.assertIn("status", result)
        
        self.assertTrue(0.0 <= result["anomaly_score"] <= 1.0)
        self.assertTrue(0.0 <= result["health_score"] <= 100.0)
        self.assertIn(result["status"], ["NORMAL", "DEGRADED", "CRITICAL"])

if __name__ == "__main__":
    unittest.main()

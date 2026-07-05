import unittest
import torch
import numpy as np
import os
from models.rul_prediction.model import LSTMRULRegressor, RULEnsemble

class TestRULPrediction(unittest.TestCase):
    def test_lstm_forward_shape(self):
        batch_size = 4
        seq_len = 30
        input_dim = 21
        model = LSTMRULRegressor(input_dim=input_dim)
        
        x = torch.randn(batch_size, seq_len, input_dim)
        y = model(x)
        self.assertEqual(y.shape, (batch_size, 1))

    def test_ensemble_inference(self):
        ensemble = RULEnsemble(lstm_path="invalid_path.pt", xgb_path="invalid_path.json")
        dummy_window = np.random.rand(30, 24)
        
        result = ensemble.predict(dummy_window)
        
        self.assertIn("lstm_rul", result)
        self.assertIn("xgb_rul", result)
        self.assertIn("ensemble_rul", result)
        self.assertIn("current_health", result)
        self.assertIn("failure_probability", result)
        self.assertIn("risk", result)
        
        self.assertGreaterEqual(result["ensemble_rul"], 0.0)
        self.assertTrue(0.0 <= result["current_health"] <= 100.0)
        self.assertTrue(0.0 <= result["failure_probability"] <= 1.0)
        self.assertIn(result["risk"], ["Low", "Medium", "High"])

if __name__ == "__main__":
    unittest.main()

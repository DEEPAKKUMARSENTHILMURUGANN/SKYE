import unittest
import numpy as np
from sensor_simulator.simulator import AircraftSensorSimulator
from edge_processing.processing import EdgeAIProcessor

class TestEdgeAIProcessor(unittest.TestCase):
    def setUp(self):
        self.simulator = AircraftSensorSimulator()
        self.processor = EdgeAIProcessor(window_size=30)

    def test_processing_pipeline(self):
        telemetry = self.simulator.get_next_telemetry()
        processed = self.processor.process_telemetry(telemetry)
        self.assertIn("raw_flat", processed)
        self.assertIn("filtered_flat", processed)
        self.assertIn("engineered_flat", processed)
        self.assertIn("normalized_flat", processed)
        self.assertIn("heuristic_health", processed)
        self.assertEqual(len(self.processor.feature_keys), 23)

    def test_sliding_window_shape(self):
        window_arr = self.processor.get_sliding_window_array()
        self.assertEqual(window_arr.shape, (30, 23))
        for _ in range(10):
            telemetry = self.simulator.get_next_telemetry()
            self.processor.process_telemetry(telemetry)
            
        window_arr = self.processor.get_sliding_window_array()
        self.assertEqual(window_arr.shape, (30, 23))
        self.assertTrue(np.all(window_arr >= 0.0))
        self.assertTrue(np.all(window_arr <= 1.0))

    def test_noise_filtering(self):
        raw_seq = [2.0, 10.0, 2.0]
        for val in raw_seq:
            raw_data = {"engine_vibration": val}
            filt_data = self.processor.filter_noise(raw_data)
            self.assertLess(filt_data["engine_vibration"], 10.0)

    def test_heuristic_health(self):
        telemetry = self.simulator.get_next_telemetry()
        telemetry["hydraulic"]["hydraulic_pressure"] = 2000.0
        telemetry["hydraulic"]["pump_efficiency"] = 60.0
        
        processed = self.processor.process_telemetry(telemetry)
        healths = processed["heuristic_health"]
        
        self.assertLess(healths["hydraulic"], 100.0)
        self.assertLess(healths["overall"], 100.0)

if __name__ == "__main__":
    unittest.main()

import unittest
from sensor_simulator.simulator import AircraftSensorSimulator

class TestAircraftSensorSimulator(unittest.TestCase):
    def setUp(self):
        self.simulator = AircraftSensorSimulator()

    def test_initial_state(self):
        self.assertEqual(self.simulator.mode, "NORMAL")
        self.assertEqual(len(self.simulator.active_faults), 0)
        self.assertEqual(self.simulator.step_count, 0)

    def test_phase_transitions(self):
        self.assertEqual(self.simulator.get_flight_phase(0), "PRE-FLIGHT")
        self.assertEqual(self.simulator.get_flight_phase(10), "PRE-FLIGHT")
        self.assertEqual(self.simulator.get_flight_phase(20), "TAKEOFF")
        self.assertEqual(self.simulator.get_flight_phase(50), "CRUISE")
        self.assertEqual(self.simulator.get_flight_phase(100), "LANDING")
        self.assertEqual(self.simulator.get_flight_phase(115), "TAXI")

    def test_telemetry_schema(self):
        telemetry = self.simulator.get_next_telemetry()
        self.assertIn("timestamp", telemetry)
        self.assertIn("step", telemetry)
        self.assertIn("flight_phase", telemetry)
        self.assertIn("engine", telemetry)
        self.assertIn("hydraulic", telemetry)
        self.assertIn("structural", telemetry)
        self.assertIn("environmental", telemetry)
        engine = telemetry["engine"]
        self.assertIn("engine_vibration", engine)
        self.assertIn("egt", engine)
        self.assertIn("oil_pressure", engine)
        self.assertIn("n1_speed", engine)
        self.assertIn("n2_speed", engine)
        self.assertIn("fuel_flow", engine)
        hydraulic = telemetry["hydraulic"]
        self.assertIn("hydraulic_pressure", hydraulic)
        self.assertIn("pump_efficiency", hydraulic)
        self.assertIn("actuator_response", hydraulic)

    def test_mode_transitions(self):
        self.assertTrue(self.simulator.set_mode("DEGRADATION"))
        self.assertEqual(self.simulator.mode, "DEGRADATION")
        
        self.assertTrue(self.simulator.set_mode("SUDDEN_ANOMALY"))
        self.assertEqual(self.simulator.mode, "SUDDEN_ANOMALY")
        
        self.assertFalse(self.simulator.set_mode("INVALID_MODE"))

    def test_fault_injection(self):
        self.simulator.inject_fault("hydraulic_leak")
        self.assertEqual(self.simulator.mode, "FAULT_INJECTION")
        self.assertIn("hydraulic_leak", self.simulator.active_faults)
        
        telemetry = self.simulator.get_next_telemetry()
        self.assertLess(telemetry["hydraulic"]["hydraulic_pressure"], 2800.0)

    def test_degradation_effect(self):
        self.simulator.set_mode("DEGRADATION")
        vibe_initial = self.simulator.get_next_telemetry()["engine"]["engine_vibration"]
        for _ in range(30):
            telemetry = self.simulator.get_next_telemetry()
            
        vibe_later = telemetry["engine"]["engine_vibration"]
        self.assertGreater(self.simulator.degradation_steps, 0)

if __name__ == "__main__":
    unittest.main()

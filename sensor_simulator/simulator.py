import time
import random
import numpy as np

class AircraftSensorSimulator:
    """
    SKYE Aircraft Sensor Simulation Engine.
    Generates realistic, physics-informed real-time flight data telemetry.
    Supports Normal, Degradation, Fault Injection, and Sudden Anomaly modes.
    """
    def __init__(self, update_interval_seconds=1.0):
        self.update_interval = update_interval_seconds
        self.reset()

    def reset(self):
        """Resets the simulator state for a new flight cycle."""
        self.start_time = time.time()
        self.step_count = 0
        self.flight_cycle = 1250
        self.manual_phase = None
        self.max_steps = 120
        self.phase_definitions = [
            (0, 15, "PRE-FLIGHT"),
            (15, 35, "TAKEOFF"),
            (35, 95, "CRUISE"),
            (95, 110, "LANDING"),
            (110, 120, "TAXI")
        ]
        
        self.mode = "NORMAL" 
        self.active_faults = set()
        self.degradation_steps = 0
        self.fault_steps = 0
        self.anomaly_steps = 0

    def set_mode(self, mode: str):
        """Sets the operating mode of the simulator."""
        valid_modes = ["NORMAL", "DEGRADATION", "FAULT_INJECTION", "SUDDEN_ANOMALY"]
        if mode in valid_modes:
            self.mode = mode
            if mode == "NORMAL":
                self.active_faults.clear()
                self.degradation_steps = 0
                self.fault_steps = 0
                self.anomaly_steps = 0
            return True
        return False

    def inject_fault(self, fault_type: str):
        """Injects a specific fault type."""
        self.active_faults.add(fault_type)
        if self.mode == "NORMAL":
            self.mode = "FAULT_INJECTION"

    def get_flight_phase(self, step):
        """Returns the current flight phase based on step count."""
        if self.manual_phase is not None:
            return self.manual_phase
        cyclic_step = step % self.max_steps
        for start, end, phase in self.phase_definitions:
            if start <= cyclic_step < end:
                return phase
        return "TAXI"

    def get_next_telemetry(self):
        """
        Generates the next state packet of telemetry data.
        Increments the state and computes values.
        """
        step = self.step_count
        phase = self.get_flight_phase(step)
        if step > 0 and step % self.max_steps == 0:
            self.flight_cycle += 1
        if self.mode == "DEGRADATION":
            self.degradation_steps += 1
        elif self.mode == "FAULT_INJECTION":
            self.fault_steps += 1
        elif self.mode == "SUDDEN_ANOMALY":
            self.anomaly_steps += 1
        cyclic_step = step % self.max_steps
        if phase == "PRE-FLIGHT":
            alt = 0.0
            speed = 0.0
        elif phase == "TAKEOFF":
            t_ratio = (cyclic_step - 15) / 20.0
            alt = t_ratio * 30000.0
            speed = t_ratio * 250.0
        elif phase == "CRUISE":
            t_ratio = (cyclic_step - 35) / 60.0
            alt = 30000.0 + 150.0 * np.sin(cyclic_step / 5.0)
            speed = 460.0 + 10.0 * np.cos(cyclic_step / 10.0)
        elif phase == "LANDING":
            t_ratio = (cyclic_step - 95) / 15.0
            alt = (1.0 - t_ratio) * 30000.0
            speed = 250.0 - t_ratio * 240.0
        else:
            alt = 0.0
            speed = 10.0
        if phase == "PRE-FLIGHT":
            vibe_base, egt_base, oil_base, n1_base, n2_base, fuel_base = 0.5, 350.0, 42.0, 20.0, 55.0, 0.25
        elif phase == "TAKEOFF":
            vibe_base, egt_base, oil_base, n1_base, n2_base, fuel_base = 2.6, 730.0, 62.0, 95.0, 98.0, 2.40
        elif phase == "CRUISE":
            vibe_base, egt_base, oil_base, n1_base, n2_base, fuel_base = 1.7, 610.0, 56.0, 88.0, 92.0, 1.35
        elif phase == "LANDING":
            vibe_base, egt_base, oil_base, n1_base, n2_base, fuel_base = 2.1, 570.0, 50.0, 58.0, 72.0, 0.85
        else:
            vibe_base, egt_base, oil_base, n1_base, n2_base, fuel_base = 0.6, 380.0, 44.0, 25.0, 58.0, 0.35
        hyd_press_base = 3000.0
        pump_perf_base = 98.0
        act_resp_base = 0.12
        if phase == "PRE-FLIGHT":
            stress_base, load_base, acc_x, acc_y, acc_z = 25.0, 1.0, 0.0, 0.0, 1.0
        elif phase == "TAKEOFF":
            stress_base, load_base, acc_x, acc_y, acc_z = 210.0, 1.22, 0.35, 0.02, 1.20
        elif phase == "CRUISE":
            stress_base, load_base, acc_x, acc_y, acc_z = 110.0, 1.00, 0.0, 0.01, 1.00
        elif phase == "LANDING":
            if 97 <= cyclic_step <= 100:
                stress_base, load_base, acc_x, acc_y, acc_z = 280.0, 1.45, -0.45, 0.05, 1.40
            else:
                stress_base, load_base, acc_x, acc_y, acc_z = 160.0, 1.10, -0.25, 0.02, 1.10
        else:
            stress_base, load_base, acc_x, acc_y, acc_z = 35.0, 1.00, 0.01, 0.0, 1.00
        cabin_press_base = 14.7 - (alt / 30000.0) * 3.3
        
        if phase == "PRE-FLIGHT":
            bleed_temp_base, ac_perf_base = 110.0, 98.0
        elif phase == "TAKEOFF":
            bleed_temp_base, ac_perf_base = 205.0, 95.0
        elif phase == "CRUISE":
            bleed_temp_base, ac_perf_base = 192.0, 96.0
        elif phase == "LANDING":
            bleed_temp_base, ac_perf_base = 165.0, 97.0
        else:
            bleed_temp_base, ac_perf_base = 125.0, 98.0
        vibe_noise = np.random.normal(0, 0.06)
        egt_noise = np.random.normal(0, 1.8)
        oil_noise = np.random.normal(0, 0.4)
        n_noise = np.random.normal(0, 0.15)
        fuel_noise = np.random.normal(0, 0.02)
        hyd_noise = np.random.normal(0, 12.0)
        pump_noise = np.random.normal(0, 0.3)
        act_noise = np.random.normal(0, 0.005)
        stress_noise = np.random.normal(0, 4.0)
        load_noise = np.random.normal(0, 0.015)
        acc_noise = np.random.normal(0, 0.01)
        cabin_noise = np.random.normal(0, 0.02)
        bleed_noise = np.random.normal(0, 0.8)
        ac_noise = np.random.normal(0, 0.2)
        vibe = vibe_base + vibe_noise
        egt = egt_base + egt_noise
        oil_pressure = oil_base + oil_noise
        n1_speed = n1_base + n_noise
        n2_speed = n2_base + n_noise
        fuel_flow = fuel_base + fuel_noise
        
        hydraulic_pressure = hyd_press_base + hyd_noise
        pump_efficiency = pump_perf_base + pump_noise
        actuator_response = act_resp_base + act_noise
        
        airframe_stress = stress_base + stress_noise
        wing_load = load_base + load_noise
        accel_x = acc_x + acc_noise
        accel_y = acc_y + acc_noise
        accel_z = acc_z + acc_noise
        
        cabin_pressure = cabin_press_base + cabin_noise
        bleed_temp = bleed_temp_base + bleed_noise
        ac_efficiency = ac_perf_base + ac_noise
        if self.mode == "DEGRADATION" or "bearing_wear" in self.active_faults:
            t = self.degradation_steps
            vibe += 0.018 * (t ** 1.15)
            egt += 0.45 * t
            fuel_flow += 0.003 * t
            pump_efficiency -= 0.08 * t
            actuator_response += 0.0015 * t
            hydraulic_pressure -= 1.5 * t
            ac_efficiency -= 0.06 * t
        if self.mode == "FAULT_INJECTION" or "hydraulic_leak" in self.active_faults:
            hydraulic_pressure -= 750.0
            pump_efficiency -= 20.0
            actuator_response += 0.25
            
        if "ecs_bleed_valve_fail" in self.active_faults:
            bleed_temp += 45.0
            ac_efficiency -= 15.0
            cabin_pressure -= 0.6
        if self.mode == "SUDDEN_ANOMALY" or "compressor_surge" in self.active_faults:
            t = self.anomaly_steps
            osc_freq = 0.8
            vibe += 1.8 + 0.4 * np.sin(t * osc_freq)
            egt += 50.0 * np.sin(t * osc_freq)
            n1_speed += 8.0 * np.sin(t * osc_freq)
            n2_speed += 6.0 * np.cos(t * osc_freq)
            fuel_flow += 0.3 * np.sin(t * osc_freq)
            
        if "wing_structural_damage" in self.active_faults:
            airframe_stress += 250.0 + 80.0 * np.random.randn()
            wing_load += 0.45 * np.random.randn()
        vibe = max(0.0, vibe)
        egt = max(20.0, egt)
        oil_pressure = max(0.0, oil_pressure)
        n1_speed = np.clip(n1_speed, 0.0, 115.0)
        n2_speed = np.clip(n2_speed, 0.0, 115.0)
        fuel_flow = max(0.0, fuel_flow)
        hydraulic_pressure = np.clip(hydraulic_pressure, 0.0, 3500.0)
        pump_efficiency = np.clip(pump_efficiency, 0.0, 100.0)
        actuator_response = max(0.01, actuator_response)
        airframe_stress = max(0.0, airframe_stress)
        cabin_pressure = np.clip(cabin_pressure, 0.0, 16.0)
        bleed_temp = max(10.0, bleed_temp)
        ac_efficiency = np.clip(ac_efficiency, 0.0, 100.0)
        telemetry = {
            "timestamp": time.time(),
            "step": step,
            "flight_cycle": self.flight_cycle,
            "flight_phase": phase,
            "altitude": round(float(alt), 2),
            "speed": round(float(speed), 2),
            "engine": {
                "engine_vibration": round(float(vibe), 3),
                "egt": round(float(egt), 1),
                "oil_pressure": round(float(oil_pressure), 2),
                "n1_speed": round(float(n1_speed), 2),
                "n2_speed": round(float(n2_speed), 2),
                "fuel_flow": round(float(fuel_flow), 3)
            },
            "hydraulic": {
                "hydraulic_pressure": round(float(hydraulic_pressure), 1),
                "pump_efficiency": round(float(pump_efficiency), 2),
                "actuator_response": round(float(actuator_response), 3)
            },
            "structural": {
                "airframe_stress": round(float(airframe_stress), 1),
                "wing_load": round(float(wing_load), 3),
                "fatigue_cycles": int(self.flight_cycle),
                "accel_x": round(float(accel_x), 3),
                "accel_y": round(float(accel_y), 3),
                "accel_z": round(float(accel_z), 3)
            },
            "environmental": {
                "cabin_pressure": round(float(cabin_pressure), 2),
                "bleed_temp": round(float(bleed_temp), 1),
                "ac_efficiency": round(float(ac_efficiency), 2)
            },
            "mode": self.mode,
            "active_faults": list(self.active_faults)
        }
        
        self.step_count += 1
        return telemetry

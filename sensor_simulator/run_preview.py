import time
import json
from sensor_simulator.simulator import AircraftSensorSimulator

def format_telemetry(data):
    """Formats telemetry for readable terminal output."""
    phase = data["flight_phase"]
    mode = data["mode"]
    step = data["step"]
    
    eng = data["engine"]
    hyd = data["hydraulic"]
    struc = data["structural"]
    env = data["environmental"]
    
    out = (
        f"[{phase:^11}] Step: {step:03d} | Mode: {mode:<15} | Alt: {data['altitude']:5.0f} ft | Spd: {data['speed']:3.0f} kt\n"
        f"  |-- ENGINE: Vibe: {eng['engine_vibration']:4.2f} mm/s | EGT: {eng['egt']:5.1f} C | Oil: {eng['oil_pressure']:4.1f} psi | Fuel: {eng['fuel_flow']:4.2f} kg/s\n"
        f"  |-- HYD:    Press: {hyd['hydraulic_pressure']:6.1f} psi | Pump: {hyd['pump_efficiency']:5.1f}% | Resp: {hyd['actuator_response']:4.2f}s\n"
        f"  |-- STRUC:  Stress: {struc['airframe_stress']:5.1f} uE | Load: {struc['wing_load']:4.2f}G | Fatigue: {struc['fatigue_cycles']}\n"
        f"  \\-- ENV:    Cabin: {env['cabin_pressure']:4.1f} psi | Bleed: {env['bleed_temp']:5.1f} C | A/C: {env['ac_efficiency']:5.1f}%\n"
    )
    if data["active_faults"]:
        out += f"  [WARNING] ACTIVE FAULTS: {', '.join(data['active_faults'])}\n"
    return out

def main():
    print("=" * 70)
    print("SKYE Aircraft Sensor Simulator - Telemetry Stream Preview")
    print("=" * 70)
    
    sim = AircraftSensorSimulator(update_interval_seconds=0.1)
    
    print("\n--- Phase 1: Running Normal Flight (10 steps) ---")
    for _ in range(10):
        data = sim.get_next_telemetry()
        print(format_telemetry(data))
        time.sleep(0.05)
        
    print("\n--- Phase 2: Injecting Bearing Wear (Degradation mode, 10 steps) ---")
    sim.set_mode("DEGRADATION")
    sim.inject_fault("bearing_wear")
    for _ in range(10):
        data = sim.get_next_telemetry()
        print(format_telemetry(data))
        time.sleep(0.05)

    print("\n--- Phase 3: Injecting Sudden Hydraulic Leak (Fault mode, 10 steps) ---")
    sim.inject_fault("hydraulic_leak")
    for _ in range(10):
        data = sim.get_next_telemetry()
        print(format_telemetry(data))
        time.sleep(0.05)

    print("\n--- Phase 4: Sudden Anomaly (Compressor Surge, 10 steps) ---")
    sim.set_mode("SUDDEN_ANOMALY")
    sim.inject_fault("compressor_surge")
    for _ in range(10):
        data = sim.get_next_telemetry()
        print(format_telemetry(data))
        time.sleep(0.05)
        
    print("=" * 70)
    print("Preview complete.")
    print("=" * 70)

if __name__ == "__main__":
    main()

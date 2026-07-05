class RecommendationEngine:
    """
    Decoupled rule-based maintenance recommendation engine.
    Translates AI anomaly indicators, subsystem health metrics, and RUL predictions
    into prioritized, actionable maintenance actions for ground crews.
    """
    @staticmethod
    def generate_recommendations(health_scores: dict, anomaly_status: str, anomaly_score: float, rul_info: dict, active_faults: list) -> list:
        """
        Generates structured maintenance tasks based on real-time telemetry analytics.
        Returns:
            list of dicts, sorted by priority (1 = highest urgency)
        """
        recommendations = []
        eng_health = health_scores.get("engine", 100.0)
        rul_val = rul_info.get("ensemble_rul", 180.0)
        
        if eng_health < 80.0 or rul_val < 140.0 or "bearing_wear" in active_faults or "compressor_surge" in active_faults:
            urgency = "LOW"
            priority = 3
            diagnosis = "Minor engine efficiency degradation detected."
            action = "Monitor EGT and fuel flow rate trends during cruise. Schedule routine inspection at next C-check."
            
            if "compressor_surge" in active_faults or eng_health < 45.0:
                urgency = "CRITICAL"
                priority = 1
                diagnosis = "Severe engine operational instability (compressor surge detected)."
                action = "Inspect engine compressor guide vanes and bleed valves. Verify sensor signal lines immediately."
            elif "bearing_wear" in active_faults or rul_val < 80.0:
                urgency = "HIGH"
                priority = 2
                diagnosis = "Accelerated engine main bearing wear suspected."
                action = "Perform physical inspection of the LP rotor bearing. Drain oil filter and inspect for metallic debris."
            elif rul_val < 130.0:
                urgency = "MEDIUM"
                priority = 3
                diagnosis = "Engine remaining useful life approaching limit."
                action = "Schedule engine bearing lubrication check during next overnight maintenance window."
                
            recommendations.append({
                "subsystem": "Engine",
                "diagnosis": diagnosis,
                "health": eng_health,
                "rul": rul_val,
                "urgency": urgency,
                "priority": priority,
                "action": action,
                "confidence": round(0.70 + 0.28 * (1.0 - (eng_health / 100.0)), 2)
            })
        hyd_health = health_scores.get("hydraulic", 100.0)
        if hyd_health < 85.0 or "hydraulic_leak" in active_faults:
            urgency = "MEDIUM"
            priority = 3
            diagnosis = "Hydraulic pressure drift or pump efficiency wear."
            action = "Examine hydraulic filter indices and pressure transducer calibration curves."
            
            if "hydraulic_leak" in active_faults or hyd_health < 50.0:
                urgency = "CRITICAL"
                priority = 1
                diagnosis = "Rapid pressure loss (suspected actuator seal rupture/hydraulic leak)."
                action = "Ground aircraft. Inspect hydraulic lines, primary actuator chambers, and pump manifold seals immediately."
            elif hyd_health < 75.0:
                urgency = "HIGH"
                priority = 2
                diagnosis = "Hydraulic pump efficiency below nominal operating range."
                action = "Inspect hydraulic fluid reservoir level and purge air from control surface actuators."
                
            recommendations.append({
                "subsystem": "Hydraulic",
                "diagnosis": diagnosis,
                "health": hyd_health,
                "rul": "N/A",
                "urgency": urgency,
                "priority": priority,
                "action": action,
                "confidence": round(0.75 + 0.23 * (1.0 - (hyd_health / 100.0)), 2)
            })
        struc_health = health_scores.get("structural", 100.0)
        fatigue = health_scores.get("fatigue_cycles", 1250)
        
        if struc_health < 80.0 or "wing_structural_damage" in active_faults or fatigue > 1300:
            urgency = "LOW"
            priority = 4
            diagnosis = "Structural fatigue accumulator alert."
            action = "Log cumulative load cycles in airframe registry. Schedule routine ultrasonic NDT of main wing spar."
            
            if "wing_structural_damage" in active_faults or struc_health < 50.0:
                urgency = "CRITICAL"
                priority = 1
                diagnosis = "Wing main structure stress limit exceeded (suspected structural micro-fracture)."
                action = "Immediate structural audit required. Conduct eddy current and ultrasonic inspection on wing root rib-joints."
            elif struc_health < 75.0:
                urgency = "HIGH"
                priority = 2
                diagnosis = "Abnormal airframe stress profile observed during flight phases."
                action = "Inspect wing-load accelerometer wiring. Inspect wing panel fastening torque values."
                
            recommendations.append({
                "subsystem": "Structural",
                "diagnosis": diagnosis,
                "health": struc_health,
                "rul": "N/A",
                "urgency": urgency,
                "priority": priority,
                "action": action,
                "confidence": round(0.80 + 0.18 * (1.0 - (struc_health / 100.0)), 2)
            })
        env_health = health_scores.get("environmental", 100.0)
        if env_health < 80.0 or "ecs_bleed_valve_fail" in active_faults:
            urgency = "MEDIUM"
            priority = 3
            diagnosis = "Environmental control system bleed air temperature drift."
            action = "Check cabin air recirculating fan filters. Calibrate zone temperature controllers."
            
            if "ecs_bleed_valve_fail" in active_faults or env_health < 50.0:
                urgency = "HIGH"
                priority = 2
                diagnosis = "Bleed air pressure valve malfunction (ecs_bleed_valve_fail)."
                action = "Inspect bleed air regulation valve actuator. Confirm valve responds to manual override inputs."
                
            recommendations.append({
                "subsystem": "Environmental",
                "diagnosis": diagnosis,
                "health": env_health,
                "rul": "N/A",
                "urgency": urgency,
                "priority": priority,
                "action": action,
                "confidence": round(0.70 + 0.25 * (1.0 - (env_health / 100.0)), 2)
            })
        recommendations.sort(key=lambda x: (x["priority"], x["health"]))
        return recommendations

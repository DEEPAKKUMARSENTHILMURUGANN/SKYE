# SKYE On-Aircraft Edge AI Diagnostics Platform

SKYE is an advanced, physics-informed synthetic telemetry generator and edge AI analytics platform for next-generation aircraft health monitoring. It integrates real-time telemetry simulation, edge feature engineering, LSTM Autoencoder-based anomaly detection, sequence-based RUL prediction, and a rule-based maintenance recommendation engine.

---

## Architecture Overview

SKYE is structured into modular components:

1. **Aircraft Sensor Simulation Engine (`sensor_simulator/`)**: Computes high-fidelity, physics-informed synthetic telemetry representing flight phases (Takeoff, Cruise, Landing) and operational modes (Normal, Degradation, Fault Injection, Sudden Anomaly).
2. **Edge AI Processing Pipeline (`edge_processing/`)**: Noise filtering (EMA), normalization, feature extraction, and sliding window management.
3. **AI Anomaly Detection Model (`models/anomaly_detection/`)**: LSTM Autoencoder that reconstructs multivariate telemetry sequences. High reconstruction loss flags anomalies.
4. **Remaining Useful Life (RUL) Prediction Model (`models/rul_prediction/`)**: Hybrid model (LSTM and XGBoost ensemble) trained to predict turbofan engine RUL.
5. **Edge Deployment Optimization (`utils/optimize.py`)**: Quantizes the LSTM model to INT8 to reduce memory footprint and latency.
6. **Maintenance Recommendation Engine (`utils/recommendation.py`)**: Evaluates ML outputs and translates them into actionable maintenance tasks.
7. **Real-Time Dash Dashboard (`dashboard/app.py`)**: A dark-theme aviation-style monitoring suite displaying live graphs, health states, RUL indicators, and priority maintenance tasks.
8. **Federated Learning Simulation (`federated_learning/`)**: Simulates privacy-preserving cooperative training across three aircraft edge nodes using FedAvg.

---

## Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Machine Learning Models
```bash
# 1. Generate synthetic NASA CMAPSS dataset & train RUL models
python -m datasets.generate_cmapss
python -m models.rul_prediction.train

# 2. Train the LSTM Autoencoder anomaly detector
python -m models.anomaly_detection.train
```

### 3. Run Optimization & Benchmarking
```bash
python -m utils.optimize
```

### 4. Run Federated Learning Simulation
```bash
python -m federated_learning.federated
```

### 5. Launch the Real-Time Dashboard
```bash
python -m dashboard.app
```
Then navigate to [http://127.0.0.1:8050](http://127.0.0.1:8050) in your browser.

---

## Running Verification Tests
To run all unit tests for the platform, execute:
```bash
python -m unittest discover -s . -p "test_*.py"
```


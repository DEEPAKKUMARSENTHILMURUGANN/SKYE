"""
SKYE Federated Learning Simulation Engine
========================================
Simulates cooperative training of the Anomaly Detection LSTM Autoencoder
across three independent aircraft edge nodes (Aircraft 1, 2, and 3)
using the Federated Averaging (FedAvg) algorithm.
Demonstrates model convergence without centralizing raw telemetry data.
"""

import copy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sensor_simulator.simulator import AircraftSensorSimulator
from edge_processing.processing import EdgeAIProcessor
from models.anomaly_detection.model import LSTMAutoencoder

class AircraftEdgeNode:
    """
    Represents an individual aircraft's onboard edge computing node.
    Maintains local telemetry data and performs local model updates.
    """
    def __init__(self, node_id: int, seed: int):
        self.node_id = node_id
        self.seed = seed
        self.local_data = None
        self._generate_local_data()

    def _generate_local_data(self):
        """Generates node-specific local flight telemetry data with unique noise signatures."""
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        
        sim = AircraftSensorSimulator()
        processor = EdgeAIProcessor(window_size=30)
        num_cycles = 2 + (self.node_id % 2)
        total_steps = num_cycles * sim.max_steps
        if self.node_id == 1:
            sim.set_mode("DEGRADATION")
            sim.inject_fault("bearing_wear")
        elif self.node_id == 2:
            sim.set_mode("FAULT_INJECTION")
            sim.inject_fault("hydraulic_leak")
        else:
            sim.set_mode("SUDDEN_ANOMALY")
            sim.inject_fault("compressor_surge")
            
        windows = []
        for _ in range(total_steps):
            telemetry = sim.get_next_telemetry()
            processor.process_telemetry(telemetry)
            window = processor.get_sliding_window_array()
            windows.append(window)
            
        self.local_data = torch.tensor(np.array(windows), dtype=torch.float32)
        print(f"[Node {self.node_id}] Generated {len(self.local_data)} telemetry sequence windows.")

    def train_local_model(self, global_state_dict: dict, epochs: int = 2, lr: float = 0.005) -> dict:
        """
        Loads the global model weights, trains locally on private telemetry,
        and returns the updated model state dictionary.
        """
        model = LSTMAutoencoder(seq_len=30, input_dim=23)
        model.load_state_dict(global_state_dict)
        model.train()
        
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        dataset_size = len(self.local_data)
        batch_size = 16
        
        for epoch in range(epochs):
            permutation = torch.randperm(dataset_size)
            for i in range(0, dataset_size, batch_size):
                indices = permutation[i:i+batch_size]
                batch = self.local_data[indices]
                
                optimizer.zero_grad()
                reconstructed = model(batch)
                loss = criterion(reconstructed, batch)
                loss.backward()
                optimizer.step()
                
        return model.state_dict()


class FederatedAggregator:
    """
    Global Server/Aggregator that coordinates the federated learning process.
    Aggregates local updates using FedAvg and evaluates the global model.
    """
    def __init__(self):
        self.global_model = LSTMAutoencoder(seq_len=30, input_dim=23)
        self.validation_data = None
        self._generate_validation_data()

    def _generate_validation_data(self):
        """Generates a separate validation dataset to evaluate global model convergence."""
        np.random.seed(999)
        torch.manual_seed(999)
        
        sim = AircraftSensorSimulator()
        processor = EdgeAIProcessor(window_size=30)
        sim.set_mode("NORMAL")
        total_steps = 2 * sim.max_steps
        
        windows = []
        for _ in range(total_steps):
            telemetry = sim.get_next_telemetry()
            processor.process_telemetry(telemetry)
            window = processor.get_sliding_window_array()
            windows.append(window)
            
        self.validation_data = torch.tensor(np.array(windows), dtype=torch.float32)

    def evaluate_global_model(self) -> float:
        """Computes reconstruction MSE loss of global model on validation data."""
        self.global_model.eval()
        criterion = nn.MSELoss()
        with torch.no_grad():
            reconstructed = self.global_model(self.validation_data)
            val_loss = criterion(reconstructed, self.validation_data).item()
        return val_loss

    def aggregate(self, local_state_dicts: list) -> dict:
        """
        Performs Federated Averaging (FedAvg) aggregation over a list of local state dicts.
        Updates the global model weights.
        """
        aggregated_dict = copy.deepcopy(local_state_dicts[0])
        for key in aggregated_dict.keys():
            for i in range(1, len(local_state_dicts)):
                aggregated_dict[key] += local_state_dicts[i][key]
            aggregated_dict[key] = aggregated_dict[key] / len(local_state_dicts)
            
        self.global_model.load_state_dict(aggregated_dict)
        return aggregated_dict


def run_federated_simulation(rounds: int = 5, local_epochs: int = 2):
    print("=" * 70)
    print("SKYE Collaborative Edge Intelligence: Federated Learning Simulation")
    print("=" * 70)
    print("\n[Global Aggregator] Initializing global model & validation set...")
    aggregator = FederatedAggregator()
    
    print("\n[Edge Nodes] Initializing aircraft nodes with local private datasets...")
    nodes = [
        AircraftEdgeNode(node_id=1, seed=101),
        AircraftEdgeNode(node_id=2, seed=202),
        AircraftEdgeNode(node_id=3, seed=303)
    ]
    initial_loss = aggregator.evaluate_global_model()
    print(f"\nRound 0: Initial Global Model Validation Loss: {initial_loss:.6f}")
    
    history = [initial_loss]
    for r in range(1, rounds + 1):
        print(f"\n--- Federated Round {r}/{rounds} ---")
        
        local_updates = []
        global_weights = aggregator.global_model.state_dict()
        for node in nodes:
            print(f"  * Node {node.node_id} downloading global weights & training locally...")
            local_weights = node.train_local_model(global_weights, epochs=local_epochs)
            local_updates.append(local_weights)
        print("  * Aggregating client weights via FedAvg...")
        aggregator.aggregate(local_updates)
        round_loss = aggregator.evaluate_global_model()
        print(f"  * Round {r} Completed | Validation Loss: {round_loss:.6f}")
        history.append(round_loss)
        
    print("\n" + "=" * 50)
    print("Federated Learning Convergence Report")
    print("-" * 50)
    print(f"Initial Validation Loss (Round 0): {history[0]:.6f}")
    print(f"Final Validation Loss (Round {rounds}): {history[-1]:.6f}")
    reduction = ((history[0] - history[-1]) / history[0]) * 100.0
    print(f"Total Loss Reduction:              {reduction:.2f}%")
    print("=" * 50)
    assert history[-1] < history[0], "Federated learning failed to reduce validation loss."
    print("Collaborative training validated: Global model improved without centralizing raw telemetry data!")
    
if __name__ == "__main__":
    run_federated_simulation()

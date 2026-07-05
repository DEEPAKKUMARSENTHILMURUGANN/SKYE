import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sensor_simulator.simulator import AircraftSensorSimulator
from edge_processing.processing import EdgeAIProcessor
from models.anomaly_detection.model import LSTMAutoencoder

def generate_training_data(num_cycles=4):
    """Generates synthetic normal flight telemetry and processes it into sliding windows."""
    sim = AircraftSensorSimulator()
    processor = EdgeAIProcessor(window_size=30)
    sim.set_mode("NORMAL")
    
    total_steps = num_cycles * sim.max_steps
    dataset = []
    
    print(f"Generating {total_steps} telemetry points of normal flight...")
    for _ in range(total_steps):
        telemetry = sim.get_next_telemetry()
        processed = processor.process_telemetry(telemetry)
        window = processor.get_sliding_window_array()
        dataset.append(window)
        
    dataset = np.array(dataset)
    return dataset

def main():
    torch.manual_seed(42)
    np.random.seed(42)
    train_data = generate_training_data(num_cycles=4)
    train_tensor = torch.tensor(train_data, dtype=torch.float32)
    seq_len = 30
    input_dim = 23
    model = LSTMAutoencoder(seq_len=seq_len, input_dim=input_dim, hidden_dim=16, latent_dim=8)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    
    epochs = 20
    batch_size = 32
    
    print("Training Anomaly Detector LSTM Autoencoder...")
    model.train()
    
    dataset_size = len(train_tensor)
    for epoch in range(epochs):
        permutation = torch.randperm(dataset_size)
        epoch_loss = 0.0
        
        for i in range(0, dataset_size, batch_size):
            indices = permutation[i:i+batch_size]
            batch = train_tensor[indices]
            
            optimizer.zero_grad()
            reconstructed = model(batch)
            loss = criterion(reconstructed, batch)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * len(batch)
            
        epoch_loss /= dataset_size
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:02d}/{epochs:02d} | Loss: {epoch_loss:.6f}")
    model.eval()
    losses = []
    with torch.no_grad():
        for i in range(dataset_size):
            sample = train_tensor[i:i+1]
            reconstructed = model(sample)
            loss = criterion(reconstructed, sample)
            losses.append(loss.item())
            
    losses = np.array(losses)
    threshold = float(np.percentile(losses, 98.0))
    print(f"Reconstruction Loss Metrics - Mean: {losses.mean():.6f} | Max: {losses.max():.6f}")
    print(f"Established 98th-percentile anomaly threshold: {threshold:.6f}")
    os.makedirs("trained_models", exist_ok=True)
    save_path = "trained_models/anomaly_detector.pt"
    
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "threshold": threshold,
        "input_dim": input_dim,
        "seq_len": seq_len
    }
    torch.save(checkpoint, save_path)
    print(f"Model and metadata successfully saved to: {save_path}")

if __name__ == "__main__":
    main()

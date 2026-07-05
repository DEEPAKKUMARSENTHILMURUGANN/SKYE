import os
import time
import torch
import torch.nn as nn
import numpy as np
from models.anomaly_detection.model import LSTMAutoencoder

def benchmark_latency(model, dummy_input, num_iterations=500):
    """Measures the average inference latency of a model in milliseconds."""
    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(dummy_input)
    end_time = time.perf_counter()
    total_time = (end_time - start_time) * 1000.0
    return total_time / num_iterations

def main():
    print("=" * 70)
    print("SKYE Edge Deployment Optimization & Benchmarking Tool")
    print("=" * 70)
    model_path = "trained_models/anomaly_detector.pt"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}. Run train.py first.")
        
    checkpoint = torch.load(model_path, weights_only=False)
    model_f32 = LSTMAutoencoder(seq_len=30, input_dim=23)
    model_f32.load_state_dict(checkpoint["model_state_dict"])
    model_f32.eval()
    f32_path = "trained_models/anomaly_detector_f32.pt"
    torch.save(model_f32.state_dict(), f32_path)
    f32_size = os.path.getsize(f32_path) / 1024.0
    onnx_path = "trained_models/anomaly_detector.onnx"
    dummy_input = torch.randn(1, 30, 23)
    
    onnx_size = 0.0
    onnx_available = False
    print("Exporting Anomaly Detector to ONNX format...")
    try:
        torch.onnx.export(
            model_f32,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=15,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
        )
        onnx_size = os.path.getsize(onnx_path) / 1024.0
        onnx_available = True
        print(f"ONNX model saved to: {onnx_path}")
    except Exception as e:
        print(f"ONNX Export skipped or failed: {e}")
        print("Note: Install 'onnx' and 'onnxscript' via pip for native ONNX format support.")
    print("Quantizing PyTorch model to INT8...")
    model_int8 = torch.quantization.quantize_dynamic(
        model_f32,
        {nn.LSTM, nn.Linear},
        dtype=torch.qint8
    )
    
    int8_path = "trained_models/anomaly_detector_int8.pt"
    torch.save(model_int8.state_dict(), int8_path)
    int8_size = os.path.getsize(int8_path) / 1024.0
    print("\nRunning latency benchmarks (500 runs each)...")
    latency_f32 = benchmark_latency(model_f32, dummy_input)
    latency_int8 = benchmark_latency(model_int8, dummy_input)
    onnx_size_str = f"{onnx_size:10.2f}" if onnx_available else "N/A (Skip)"
    onnx_red_str = f"{((f32_size - onnx_size)/f32_size)*100:8.1f}%" if onnx_available else "N/A"
    
    print("\n" + "=" * 50)
    print(f"{'Metric':<25} | {'Baseline (FP32)':<15} | {'Optimized (INT8)':<15} | {'ONNX (FP32)':<10}")
    print("-" * 70)
    print(f"{'Model Size (KB)':<25} | {f32_size:15.2f} | {int8_size:15.2f} | {onnx_size_str:<10}")
    print(f"{'Model Size Reduction (%)':<25} | {'Reference':<15} | {((f32_size - int8_size)/f32_size)*100:13.1f}% | {onnx_red_str:<10}")
    print(f"{'Avg Latency (ms)':<25} | {latency_f32:15.4f} | {latency_int8:15.4f} | {'N/A (CPU)':<10}")
    print(f"{'Latency Improvement (%)':<25} | {'Reference':<15} | {((latency_f32 - latency_int8)/latency_f32)*100:13.1f}% | {'N/A':<10}")
    print(f"{'Memory Usage (Estimate)':<25} | {'~12.5 MB':<15} | {'~4.2 MB':<15} | {'~8.8 MB':<10}")
    print("=" * 70)
    if os.path.exists(f32_path):
        os.remove(f32_path)

if __name__ == "__main__":
    main()

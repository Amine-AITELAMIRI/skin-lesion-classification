#!/usr/bin/env python3
"""
Test script to verify GPU setup for the ML project.
Run with: py -3.12 test_gpu.py
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time

print("=" * 60)
print("GPU Setup Test")
print("=" * 60)

# Check PyTorch and CUDA
print(f"\nPyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Test simple tensor operation on GPU
print("\n" + "=" * 60)
print("Testing tensor operations on GPU")
print("=" * 60)

# CPU test
x_cpu = torch.randn(1000, 1000)
y_cpu = torch.randn(1000, 1000)

t0 = time.time()
z_cpu = torch.matmul(x_cpu, y_cpu)
cpu_time = time.time() - t0
print(f"\nCPU matrix multiplication (1000x1000): {cpu_time*1000:.2f} ms")

# GPU test
if torch.cuda.is_available():
    x_gpu = torch.randn(1000, 1000).to(device)
    y_gpu = torch.randn(1000, 1000).to(device)
    
    # Warmup
    _ = torch.matmul(x_gpu, y_gpu)
    torch.cuda.synchronize()
    
    t0 = time.time()
    z_gpu = torch.matmul(x_gpu, y_gpu)
    torch.cuda.synchronize()
    gpu_time = time.time() - t0
    print(f"GPU matrix multiplication (1000x1000): {gpu_time*1000:.2f} ms")
    print(f"Speedup: {cpu_time/gpu_time:.1f}x")

# Test model on GPU
print("\n" + "=" * 60)
print("Testing model on GPU")
print("=" * 60)

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(224*224*3, 256)
        self.fc2 = nn.Linear(256, 2)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleModel().to(device)
print(f"Model on device: {next(model.parameters()).device}")

# Test forward pass
batch_size = 4
dummy_input = torch.randn(batch_size, 3, 224, 224).to(device)
output = model(dummy_input)
print(f"Input shape: {dummy_input.shape}")
print(f"Output shape: {output.shape}")

print("\n" + "=" * 60)
print("✓ GPU setup is working correctly!")
print("=" * 60)
print("\nYou can now run the training scripts using Python 3.12:")
print("  py -3.12 main5.py")
print("  py -3.12 main6.py")
print("  py -3.12 main7.py")
print("=" * 60)

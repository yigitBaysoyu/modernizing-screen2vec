import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"Current GPU: {torch.cuda.current_device()}")
else:
    print("GPU not available. Running on CPU.")

# Test by creating a tensor
x = torch.randn(2, 2)
print(f"\nTensor device (default): {x.device}")

# Move tensor to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x_on_device = x.to(device)
print(f"Tensor device (after .to()): {x_on_device.device}")

import os
import torch
data_dir = "/home/ayush/Desktop/Datasets/archive"
torch.cuda.init() 
# print(os.path.join(data_dir, "Test/"))
# print(os.listdir(os.path.join(data_dir, "Test/")))
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")  # Should be True
print(f"CUDA initialized: {torch.cuda.is_initialized()}")  # Your issue
print(f"CUDA version: {torch.version.cuda}")  # Should match your system
print(f"GPU count: {torch.cuda.device_count()}")  # Should be >0
print(f"Current device: {torch.cuda.current_device()}")  # Should be 0 if GPU exists
print(f"Device name: {torch.cuda.get_device_name(0)}")  # Should print your GPU
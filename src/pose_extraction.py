#pose_extraction.py
# label-studio start

# first test if the GPU is available as pose extraction of 848 clips would be very slow on CPU
import torch
print(f"GPU available: {torch.cuda.is_available()}")
print(f"GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")

'''
GPU available: True
GPU name: NVIDIA GeForce RTX 4050 Laptop GPU
'''


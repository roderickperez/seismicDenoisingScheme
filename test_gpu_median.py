import torch
import numpy as np
import time

padded_shape = (1024, 512)
max_overlaps = 64
dtype = torch.float32

# Allocate on CPU
layers_cpu = np.random.randn(max_overlaps, *padded_shape).astype(np.float32)
# Introduce some NaNs (simulating boundary pixels)
layers_cpu[:, :128, :] = np.nan
layers_cpu[:, :, :128] = np.nan

# 1. CPU nanmedian
start = time.time()
med_cpu = np.nanmedian(layers_cpu, axis=0)
cpu_time = time.time() - start
print(f"CPU nanmedian time: {cpu_time:.4f} seconds")

# 2. GPU nanmedian using PyTorch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:", device)

if device.type == 'cuda':
    # Transfer to GPU
    start = time.time()
    layers_gpu = torch.from_numpy(layers_cpu).to(device)
    # PyTorch nanmedian
    med_gpu_t = torch.nanmedian(layers_gpu, dim=0).values
    med_gpu = med_gpu_t.cpu().numpy()
    gpu_time = time.time() - start
    print(f"GPU nanmedian time: {gpu_time:.4f} seconds")
    print("Match:", np.allclose(med_cpu, med_gpu, equal_nan=True))
else:
    print("GPU not available for benchmarking.")

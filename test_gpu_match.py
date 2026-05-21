import torch
import numpy as np

# Let's test with some random numbers and some NaNs, but no all-NaN slices
max_overlaps = 64
padded_shape = (102, 51)
layers_cpu = np.random.randn(max_overlaps, *padded_shape).astype(np.float32)

# Set some random entries to NaN
mask = np.random.rand(*layers_cpu.shape) < 0.1
layers_cpu[mask] = np.nan

# Compute CPU median
med_cpu = np.nanmedian(layers_cpu, axis=0)

# Compute GPU median
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
layers_gpu = torch.from_numpy(layers_cpu).to(device)
med_gpu_t = torch.nanmedian(layers_gpu, dim=0).values
med_gpu = med_gpu_t.cpu().numpy()

diff = np.abs(med_cpu - med_gpu)
print("Max diff:", np.max(np.nan_to_num(diff)))
print("Match (rtol=1e-5, atol=1e-5):", np.allclose(med_cpu, med_gpu, equal_nan=True, rtol=1e-5, atol=1e-5))

# Let's check all-NaN slice
layers_cpu_all_nan = np.full((max_overlaps, 2, 2), np.nan, dtype=np.float32)
layers_gpu_all_nan = torch.from_numpy(layers_cpu_all_nan).to(device)
res_cpu = np.nanmedian(layers_cpu_all_nan, axis=0)
res_gpu = torch.nanmedian(layers_gpu_all_nan, dim=0).values.cpu().numpy()
print("All-NaN CPU:", res_cpu)
print("All-NaN GPU:", res_gpu)
print("All-NaN match:", np.allclose(res_cpu, res_gpu, equal_nan=True))

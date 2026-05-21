import time
import numpy as np
import torch
from pathlib import Path
import segyio
from models.Attention_unet import AttU_Net
from model_usage_and_3d_denoising import denoise_section_2d, load_denoiser

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', device)

print("Loading model...")
model = load_denoiser('checkpoints/att_u_fine.pt', device=device)

print("Loading one inline...")
p = Path('datasets/1_Original_Seismics.sgy')
with segyio.open(str(p), strict=False, ignore_geometry=True) as f:
    # Let's collect a few traces and form an inline
    # The first inline is 100.
    il = np.array(f.attributes(segyio.TraceField.INLINE_3D)[:])
    xl = np.array(f.attributes(segyio.TraceField.CROSSLINE_3D)[:])
    uil = np.unique(il)
    uxl = np.unique(xl)
    
    # Let's find traces belonging to inline 400
    target_il = 400
    mask = (il == target_il)
    matching_indices = np.where(mask)[0]
    
    inline_data = np.zeros((len(uxl), len(f.samples)), dtype=np.float32)
    xl_map = {v: i for i, v in enumerate(uxl)}
    
    for idx in matching_indices:
        trace = f.trace[idx]
        inline_data[xl_map[xl[idx]], :] = trace

print("Inline shape:", inline_data.shape)

print("Benchmarking denoise_section_2d...")
start_time = time.time()
denoised = denoise_section_2d(inline_data, model, device=device, denoise_strength=0.3)
duration = time.time() - start_time
print(f"Time taken for one inline: {duration:.4f} seconds")

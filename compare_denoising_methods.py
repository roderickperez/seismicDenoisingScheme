import time
import numpy as np
import torch
from pathlib import Path
from models.Attention_unet import AttU_Net
from model_usage_and_3d_denoising import normalize_pct, pad_to_multiple_128, load_denoiser, load_seismic_volume
from sporco import array

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', device)

print("Loading model...")
model = load_denoiser('checkpoints/att_u_fine.pt', device=device)

# Load real seismic volume
volume = load_seismic_volume('datasets/1_Original_Seismics.sgy')
orig_inline = volume[volume.shape[0] // 2, :, :]

# Implement vectorized combine_blocks
def combine_blocks_vectorized(blks, imgsz, stpsz, fn=np.median):
    blksz = blks.shape[:-1]
    numblocks = tuple(int(np.floor((a-b)/c) + 1) for a, b, c in zip(imgsz, blksz, stpsz))
    new_shape = blksz + numblocks
    blks_reshaped = np.reshape(blks, new_shape)
    
    if fn == np.mean:
        acc = np.zeros(imgsz, dtype=blks.dtype)
        cnt = np.zeros(imgsz, dtype=blks.dtype)
        for pos in np.ndindex(numblocks):
            i, j = pos
            slices = (slice(i * stpsz[0], i * stpsz[0] + blksz[0]),
                      slice(j * stpsz[1], j * stpsz[1] + blksz[1]))
            acc[slices] += blks_reshaped[(Ellipsis,) + pos]
            cnt[slices] += 1
        return acc / cnt

    # Median path
    K_y = int(np.ceil(blksz[0] / stpsz[0]))
    K_x = int(np.ceil(blksz[1] / stpsz[1]))
    max_overlaps = K_y * K_x
    layers = np.full((max_overlaps,) + imgsz, np.nan, dtype=blks.dtype)
    for pos in np.ndindex(numblocks):
        i, j = pos
        k = (i % K_y) * K_x + (j % K_x)
        slices = (slice(i * stpsz[0], i * stpsz[0] + blksz[0]),
                  slice(j * stpsz[1], j * stpsz[1] + blksz[1]))
        layers[k][slices] = blks_reshaped[(Ellipsis,) + pos]
    return np.nanmedian(layers, axis=0)

def denoise_section_2d_custom(section_2d, model, fn=np.median, use_vectorized=True):
    sec_norm, sec_lo, sec_hi = normalize_pct(section_2d)
    padded, top, bot, lf, rt = pad_to_multiple_128(sec_norm)
    blksz = (128, 128)
    stpsz = (16, 16)
    
    blocks = array.extract_blocks(padded, blksz, stpsz).transpose(2, 0, 1).astype(np.float32)
    blocks_t = torch.from_numpy(blocks).unsqueeze(1)
    loader = torch.utils.data.DataLoader(blocks_t, batch_size=64, shuffle=False)
    
    rec = []
    with torch.no_grad():
        for batch in loader:
            pred = model(batch.to(device))
            rec.append(pred.cpu())
            
    rec_blocks = torch.cat(rec, dim=0).numpy().transpose(0, 2, 3, 1).squeeze(-1)
    rec_flat = rec_blocks.transpose(1, 2, 0).reshape(np.prod(blksz), -1)
    
    if use_vectorized:
        rec_img = combine_blocks_vectorized(rec_flat.reshape(blksz + (-1,)), padded.shape, stpsz, fn)
    else:
        rec_img = array.combine_blocks(rec_flat.reshape(blksz + (-1,)), padded.shape, stpsz, fn)
        
    h_end = rec_img.shape[0] - bot if bot > 0 else rec_img.shape[0]
    w_end = rec_img.shape[1] - rt if rt > 0 else rec_img.shape[1]
    rec_crop = rec_img[top:h_end, lf:w_end]
    
    denoise_strength = 0.3
    rec_crop = (1.0 - denoise_strength) * sec_norm + denoise_strength * rec_crop
    return rec_crop * (sec_hi - sec_lo) + sec_lo

# Run comparisons
print("1. Denoising with sporco + np.median...")
t0 = time.time()
den_sporco_median = denoise_section_2d_custom(orig_inline, model, fn=np.median, use_vectorized=False)
print(f"Time: {time.time() - t0:.2f}s")

print("2. Denoising with vectorized + np.median...")
t0 = time.time()
den_vectorized_median = denoise_section_2d_custom(orig_inline, model, fn=np.median, use_vectorized=True)
print(f"Time: {time.time() - t0:.2f}s")

print("3. Denoising with vectorized + np.mean...")
t0 = time.time()
den_vectorized_mean = denoise_section_2d_custom(orig_inline, model, fn=np.mean, use_vectorized=True)
print(f"Time: {time.time() - t0:.2f}s")

# Let's compare difference
diff_med = np.abs(den_sporco_median - den_vectorized_median)
diff_mean = np.abs(den_sporco_median - den_vectorized_mean)
print("Max diff (Sporco Med vs Vectorized Med):", np.max(diff_med))
print("Max diff (Sporco Med vs Vectorized Mean):", np.max(diff_mean))
print("Mean diff (Sporco Med vs Vectorized Mean):", np.mean(diff_mean))

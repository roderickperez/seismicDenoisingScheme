#!/usr/bin/env python
# coding: utf-8

# # Seismic model usage and 3D volume denoising
# 
# This notebook does two things:
# 1. Uses the repository denoising model with training-style synthetic noisy data.
# 2. Loads a seismic input (.npy or .segy), applies the model slice-by-slice, and returns/saves a denoised result.

# In[1]:


import math
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from sporco import array
import segyio

from models.Attention_unet import AttU_Net
from degradationOperator import degradeBatch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', device)


# In[2]:


def load_denoiser(checkpoint_path='checkpoints/att_u_fine.pt', device=device):
    model = AttU_Net(img_ch=1, output_ch=1).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model

# Options: checkpoints/att_u_fine.pt or checkpoints/guided_att_u.pt
model = load_denoiser('checkpoints/att_u_fine.pt')
print('Model loaded.')


# In[3]:


def mirror_padding(image, top_padding, bottom_padding, left_padding, right_padding):
    h, w = image.shape
    new_h = h + top_padding + bottom_padding
    new_w = w + left_padding + right_padding
    padded = np.zeros((new_h, new_w), dtype=image.dtype)
    padded[top_padding:top_padding + h, left_padding:left_padding + w] = image

    if top_padding > 0:
        padded[:top_padding, left_padding:left_padding + w] = image[:top_padding][::-1]
    if bottom_padding > 0:
        padded[top_padding + h:, left_padding:left_padding + w] = image[-bottom_padding:][::-1]
    if left_padding > 0:
        padded[top_padding:top_padding + h, :left_padding] = image[:, :left_padding][:, ::-1]
    if right_padding > 0:
        padded[top_padding:top_padding + h, left_padding + w:] = image[:, -right_padding:][:, ::-1]

    if top_padding > 0 and left_padding > 0:
        padded[:top_padding, :left_padding] = image[:top_padding, :left_padding][::-1, ::-1]
    if top_padding > 0 and right_padding > 0:
        padded[:top_padding, left_padding + w:] = image[:top_padding, -right_padding:][::-1, ::-1]
    if bottom_padding > 0 and left_padding > 0:
        padded[top_padding + h:, :left_padding] = image[-bottom_padding:, :left_padding][::-1, ::-1]
    if bottom_padding > 0 and right_padding > 0:
        padded[top_padding + h:, left_padding + w:] = image[-bottom_padding:, -right_padding:][::-1, ::-1]

    return padded

def pad_to_multiple_128(img):
    top = bot = lf = rt = 0
    out = img.copy()

    if out.shape[0] % 128 != 0:
        pad = math.ceil(out.shape[0] / 128) * 128 - out.shape[0]
        top = math.ceil(pad / 2)
        bot = math.floor(pad / 2)
        out = mirror_padding(out, top, bot, 0, 0)

    if out.shape[1] % 128 != 0:
        pad = math.ceil(out.shape[1] / 128) * 128 - out.shape[1]
        lf = math.ceil(pad / 2)
        rt = math.floor(pad / 2)
        out = mirror_padding(out, 0, 0, lf, rt)

    return out, top, bot, lf, rt

def normalize01(x):
    x = x.astype(np.float32)
    mn, mx = float(np.min(x)), float(np.max(x))
    if mx - mn < 1e-8:
        return np.zeros_like(x, dtype=np.float32), mn, mx
    return (x - mn) / (mx - mn), mn, mx

def normalize_pct(x, pct_low=1, pct_high=99):
    """Percentile-based normalization to [0,1].
    Clips outliers at pct_low/pct_high before scaling, so that the bulk
    of the seismic signal spans the full [0,1] range instead of being
    compressed by a few amplitude spikes."""
    x = x.astype(np.float32)
    lo = float(np.percentile(x, pct_low))
    hi = float(np.percentile(x, pct_high))
    if hi - lo < 1e-8:
        return np.zeros_like(x, dtype=np.float32), lo, hi
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0), lo, hi

def denoise_section_2d(
    section_2d,
    model,
    device=device,
    blksz=(128, 128),
    stpsz=(16, 16),
    batch_size=64,
    denoise_strength=0.5,
):
    # Percentile normalisation: maps the 1–99 pct amplitude range → [0,1]
    # This prevents a few large-amplitude outliers from compressing real
    # reflectors into a tiny sub-range where the model mistakes them for noise.
    sec_norm, sec_lo, sec_hi = normalize_pct(section_2d)
    padded, top, bot, lf, rt = pad_to_multiple_128(sec_norm)

    blocks = array.extract_blocks(padded, blksz, stpsz).transpose(2, 0, 1).astype(np.float32)
    blocks_t = torch.from_numpy(blocks).unsqueeze(1)
    loader = torch.utils.data.DataLoader(blocks_t, batch_size=batch_size, shuffle=False)

    rec = []
    with torch.no_grad():
        for batch in loader:
            pred = model(batch.to(device))
            rec.append(pred.cpu())

    rec_blocks = torch.cat(rec, dim=0).numpy().transpose(0, 2, 3, 1).squeeze(-1)
    rec_flat = rec_blocks.transpose(1, 2, 0).reshape(np.prod(blksz), -1)
    rec_img = array.combine_blocks(rec_flat.reshape(blksz + (-1,)), padded.shape, stpsz, np.median)

    h_end = rec_img.shape[0] - bot if bot > 0 else rec_img.shape[0]
    w_end = rec_img.shape[1] - rt if rt > 0 else rec_img.shape[1]
    rec_crop = rec_img[top:h_end, lf:w_end]

    denoise_strength = float(np.clip(denoise_strength, 0.0, 1.0))
    rec_crop = (1.0 - denoise_strength) * sec_norm + denoise_strength * rec_crop

    # Rescale back to original amplitude units
    if sec_hi - sec_lo < 1e-8:
        return np.full_like(rec_crop, sec_lo, dtype=np.float32)
    return (rec_crop * (sec_hi - sec_lo) + sec_lo).astype(np.float32)


# ## A) Use the model with training-style data
# This creates a noisy sample using degradeBatch from a clean patch extracted from datasets/mobileAvo.npy, then denoises it.

# In[4]:


from degradeFunctions import degradeImage

# All 16 noise types
NOISE_TYPES = [
    'gaussian', 'poisson', 'stripes', 'impulse', 'speckle',
    'corrG1', 'corrG2', 'corrG1V', 'corrG1I', 'blur',
    'streak', 'lines', 'waves', 'waves2', 's1', 's1blur',
]

def apply_single_noise(patch, noise_type):
    """Apply one named noise type to a [0,1] torch tensor patch."""
    d = degradeImage(patch.clone())
    noise_map = {
        'gaussian' : lambda: d.gaussianNoise(np.random.random(), np.random.uniform(0.35, 0.45)),
        'poisson'  : lambda: d.poissonNoise(np.random.uniform(0.3, 0.65)),
        'stripes'  : lambda: d.stripesNoise(np.random.uniform(0.2, 0.3), np.random.randint(10, 20)),
        'impulse'  : lambda: d.impulseNoise(np.random.uniform(0.04, 0.15)),
        'speckle'  : lambda: d.speckleNoise(np.random.uniform(0.15, 0.3)),
        'corrG1'   : lambda: d.convolutionG1(np.random.uniform(0.2, 0.4)),
        'corrG2'   : lambda: d.convolutionG2(np.random.uniform(0.2, 0.3)),
        'corrG1V'  : lambda: d.convolutionG1V(np.random.uniform(0.2, 0.4)),
        'corrG1I'  : lambda: d.convolutionG1I(np.random.uniform(0.25, 0.4), np.random.uniform(0.8, 1)),
        'blur'     : lambda: d.gaussianBlur(np.random.uniform(1.8, 2.2)),
        'streak'   : lambda: d.streak(np.random.randint(15, 40)),
        'lines'    : lambda: d.lines(np.random.randint(15, 40)),
        'waves'    : lambda: d.waves(np.random.uniform(0.15, 0.35),
                                     np.random.randint(1, 40), np.random.randint(40, 70)),
        'waves2'   : lambda: d.waves2(np.random.uniform(0.002, 0.02), np.random.uniform(0.002, 0.02),
                                      np.random.uniform(0.15, 0.3), np.random.uniform(0.8, 1.0),
                                      np.random.randint(0, 2)),
        's1'       : lambda: d.s1(np.random.uniform(0.23, 0.4), np.random.uniform(0.23, 0.3)),
        's1blur'   : lambda: d.s1Blur(np.random.uniform(0.25, 0.5), np.random.uniform(0.25, 0.5), 1),
    }
    return noise_map[noise_type]().numpy().astype(np.float32)

# ── Load data & pick one random patch (same patch for all noise types) ────────
clean_2d = np.load('datasets/mobileAvo.npy').astype(np.float32)
clean_2d, _, _ = normalize01(clean_2d)

h, w = clean_2d.shape
r0 = np.random.randint(0, h - 128)
c0 = np.random.randint(0, w - 128)
print(f'Selected patch at r0={r0}, c0={c0}')

clean_patch = torch.from_numpy(clean_2d[r0:r0+128, c0:c0+128])
clean_np = clean_patch.clone().float()
clean_np -= clean_np.min(); clean_np /= clean_np.max()
clean_np = clean_np.numpy()

# ── Build rows: (noise_name, noisy, denoised, diff) ─────────────────────────
rows = []
for nt in NOISE_TYPES:
    noisy = apply_single_noise(clean_patch, nt)
    den   = denoise_section_2d(noisy, model, device=device, stpsz=(16, 16),
                               batch_size=32, denoise_strength=1.0)
    diff  = clean_np - den
    rows.append((nt, noisy, den, diff))

# ── Plot ─────────────────────────────────────────────────────────────────────
n_rows = len(rows)
col_titles = ['Clean', 'Noisy', 'Denoised']

# Full [0,1] range for Clean/Noisy/Denoised
vmin, vmax = 0, 1

# n_rows image rows + 1 thin colorbar row
fig, axes = plt.subplots(
    n_rows + 1, 3, figsize=(12, n_rows * 2.2),
    gridspec_kw={'height_ratios': [10] * n_rows + [0.5], 'hspace': 0.15, 'wspace': 0.2}
)

# Column headers on first row only
for c, ct in enumerate(col_titles):
    axes[0, c].set_title(ct, fontsize=11, pad=4)

col_ims = [None] * 3
for r, (noise_name, noisy, den, diff) in enumerate(rows):
    imgs   = [clean_np, noisy, den]
    vmins  = [vmin,  vmin,  vmin]
    vmaxes = [vmax,  vmax,  vmax]
    for c, img in enumerate(imgs):
        im = axes[r, c].imshow(img, cmap='gray', vmin=vmins[c], vmax=vmaxes[c], aspect='auto')
        axes[r, c].axis('off')
        col_ims[c] = im

    # Noise type label rotated on the left of each row
    axes[r, 0].set_ylabel(noise_name, rotation=90, fontsize=9, labelpad=4, va='center')
    axes[r, 0].axis('on')
    axes[r, 0].set_yticks([]); axes[r, 0].set_xticks([])
    for spine in axes[r, 0].spines.values():
        spine.set_visible(False)

# One colorbar per column in the last row
for c, im in enumerate(col_ims):
    ticks = [0.0, 0.5, 1.0]
    fig.colorbar(im, cax=axes[-1, c], orientation='horizontal', ticks=ticks, label='Amplitude')
    axes[-1, c].tick_params(labelsize=7)

fig.suptitle(f'All noise types  |  patch r0={r0}, c0={c0}', y=0.9, fontsize=18)
plt.show()


# ## B) Load 2D/3D seismic data (.npy or .segy)
# 
# ### Why does the network require [0, 1] input — and does it work on real seismic amplitudes?
# 
# **Network normalisation requirement**
# 
# The model was trained exclusively on data normalised to the **[0, 1]** range. This is standard practice for deep learning: gradient magnitudes stay well-behaved, activations don't saturate, and the loss landscape is easier to optimise. Feeding raw seismic amplitudes (e.g. ±23 000) would be completely out-of-distribution — the outputs would be meaningless.
# 
# **The pipeline is transparent to the caller**
# 
# `denoise_section_2d` handles normalisation internally, so you can pass any raw slice and get back a result in the original amplitude units:
# 
# ```
# raw slice (e.g. ±23 000)
#   → normalize_pct()          map the 1–99 pct range → [0, 1]
#   → patch extraction + model  denoised patches still in [0, 1]
#   → rescale back              output × (sec_hi − sec_lo) + sec_lo
# → denoised slice (±23 000)   same physical units as the input
# ```
# 
# The `sec_lo` / `sec_hi` percentile anchors are stored before the model is called and used to invert the normalisation afterwards, so no amplitude information is lost.
# 
# **Why percentile normalisation instead of min/max?**
# 
# Field seismic data often has a small number of very large spikes (noise bursts, trace glitches). A min/max normalisation would compress the bulk of the reflector signal into a tiny sub-range near 0.5, so the model would perceive it as near-constant background and over-denoise real geology. Clipping at the 1st–99th percentile means the main signal always uses the full [0, 1] dynamic range.
# 

# In[5]:


def load_seismic_volume(path):
    p = Path(path)
    ext = p.suffix.lower()

    if ext == '.npy':
        arr = np.load(p).astype(np.float32)
        if arr.ndim == 2:
            arr = arr[np.newaxis, ...]
        elif arr.ndim != 3:
            raise ValueError(f'Unsupported npy ndim={arr.ndim}. Use 2D or 3D array.')
        return arr

    if ext in {'.sgy', '.segy'}:
        # Try direct cube first
        try:
            with segyio.open(str(p), strict=False, ignore_geometry=False) as f:
                cube = segyio.tools.cube(f).astype(np.float32)
                if cube.ndim == 3:
                    return cube
        except Exception:
            pass

        # Reconstruct cube from inline/xline headers (handles sparse/missing traces)
        with segyio.open(str(p), strict=False, ignore_geometry=True) as f:
            traces = segyio.tools.collect(f.trace[:]).astype(np.float32)  # [n_traces, n_samples]
            il = np.array(f.attributes(segyio.TraceField.INLINE_3D)[:])
            xl = np.array(f.attributes(segyio.TraceField.CROSSLINE_3D)[:])

            uil = np.unique(il)
            uxl = np.unique(xl)

            vol = np.full((len(uil), len(uxl), traces.shape[1]), np.nan, dtype=np.float32)
            il_map = {v: i for i, v in enumerate(uil)}
            xl_map = {v: i for i, v in enumerate(uxl)}

            for idx in range(traces.shape[0]):
                vol[il_map[il[idx]], xl_map[xl[idx]], :] = traces[idx]

            # Fill missing traces (if any) with zeros for downstream denoising
            vol = np.nan_to_num(vol, nan=0.0)
            return vol

    raise ValueError(f'Unsupported file extension: {ext}')


# In[6]:


# Use the requested SEG-Y dataset
input_path = 'datasets/1_Original_Seismics.sgy'
dataset_name = Path(input_path).name

volume = load_seismic_volume(input_path)
print('Dataset:', dataset_name)
print('Loaded full volume shape:', volume.shape)

# Middle indices from the FULL loaded volume
n_inline, n_xline, n_time = volume.shape
i_mid = n_inline // 2
x_mid = n_xline // 2
t_mid = n_time // 2

# Extract middle slices
orig_inline    = volume[i_mid, :, :]
orig_xline     = volume[:, x_mid, :]
orig_timeslice = volume[:, :, t_mid]

# denoise_strength: 1.0 = full model output, 0.0 = no change.
# 0.7 is a good starting point for field data; increase only if noise remains visible.
denoise_strength = 0.3

# Apply model to those middle slices
den_inline    = denoise_section_2d(orig_inline,    model, device=device, batch_size=64, stpsz=(16, 16), denoise_strength=denoise_strength)
den_xline     = denoise_section_2d(orig_xline,     model, device=device, batch_size=64, stpsz=(16, 16), denoise_strength=denoise_strength)
den_timeslice = denoise_section_2d(orig_timeslice, model, device=device, batch_size=64, stpsz=(16, 16), denoise_strength=denoise_strength)

diff_inline    = orig_inline    - den_inline
diff_xline     = orig_xline     - den_xline
diff_timeslice = orig_timeslice - den_timeslice

print(f'Middle indices -> inline: {i_mid}, xline: {x_mid}, timeslice: {t_mid}')
print(f'Denoise strength: {denoise_strength:.2f}')


# ### Visualisation — color scale
# 
# The same spike problem that motivates percentile normalisation for the model also affects display: using the absolute min/max as `vmin`/`vmax` would compress all reflectors into mid-gray. Instead, color limits are derived from the **1st–99th percentile** of the combined original + denoised slices, so the bulk of the seismic signal spans the full black-to-white range. The difference column gets its own tighter symmetric range computed from the actual residual values.
# 

# In[7]:


from matplotlib.patches import Rectangle

# 3x3 view: rows = [inline, xline, timeslice], cols = [original, denoised, difference]
rows = [
    ("Inline",    orig_inline,    den_inline,    diff_inline),
    ("Xline",     orig_xline,     den_xline,     diff_xline),
    ("Timeslice", orig_timeslice, den_timeslice, diff_timeslice),
]

def view_transform(img):
    return np.fliplr(img.T)

# Percentile-based color limits: use the 1-99 pct range of ORIGINAL/DENOISED slices so
# that a few amplitude spikes don't compress the bulk of the signal into mid-gray.
orig_den_arrays = [orig_inline, den_inline,
                   orig_xline,  den_xline,
                   orig_timeslice, den_timeslice]
all_vals = np.concatenate([a.ravel() for a in orig_den_arrays])
pct_abs  = max(abs(float(np.percentile(all_vals, 1))),
               abs(float(np.percentile(all_vals, 99))))
global_vmin = -pct_abs
global_vmax =  pct_abs

# Tight symmetric range for the difference columns
diff_arrays = [diff_inline, diff_xline, diff_timeslice]
all_diffs   = np.concatenate([a.ravel() for a in diff_arrays])
diff_abs    = max(abs(float(np.percentile(all_diffs, 1))),
                  abs(float(np.percentile(all_diffs, 99))))
diff_vmin, diff_vmax = -diff_abs, diff_abs

# ROI definition for the zoom (fractions of transformed image width/height)
# (x0_frac, x1_frac, y0_frac, y1_frac)
zoom_windows = {
    "Inline":    (0.35, 0.65, 0.35, 0.65),
    "Xline":     (0.35, 0.65, 0.35, 0.65),
    "Timeslice": (0.35, 0.65, 0.35, 0.65),
}

col_titles = ['Original', 'Denoised', 'Removed noise (orig - denoised)', 'Removed noise (same scale)']

# 4 rows: 3 image rows + 1 thin colorbar row, 4 columns
fig, axes = plt.subplots(
    4, 4, figsize=(22, 10),
    gridspec_kw={'height_ratios': [10, 10, 10, 0.4], 'hspace': 0.2, 'wspace': 0.25}
)
fig.suptitle(
    f"Dataset: {dataset_name} | Middle inline={i_mid}, xline={x_mid}, timeslice={t_mid}",
    fontsize=14, y=0.95
)

# Share X and Y axes row-wise for the image rows
for r in range(3):
    for c in range(1, 4):
        axes[r, c].sharex(axes[r, 0])
        axes[r, c].sharey(axes[r, 0])

# Fill image rows and add red ROI rectangles
col_ims = [None, None, None, None]
for r, (row_name, orig, den, diff) in enumerate(rows):
    imgs   = [orig, den, diff, diff]
    vmins  = [global_vmin, global_vmin, diff_vmin, global_vmin]
    vmaxes = [global_vmax, global_vmax, diff_vmax, global_vmax]

    # Compute ROI rectangle in transformed coordinates
    x0f, x1f, y0f, y1f = zoom_windows[row_name]
    vis_ref = view_transform(orig)
    h, w = vis_ref.shape
    x0 = int(np.clip(round(x0f * w), 0, w - 1))
    x1 = int(np.clip(round(x1f * w), x0 + 1, w))
    y0 = int(np.clip(round(y0f * h), 0, h - 1))
    y1 = int(np.clip(round(y1f * h), y0 + 1, h))

    for c, img in enumerate(imgs):
        im = axes[r, c].imshow(view_transform(img), cmap='gray', aspect='auto',
                               vmin=vmins[c], vmax=vmaxes[c])
        axes[r, c].add_patch(
            Rectangle((x0, y0), x1 - x0, y1 - y0,
                      edgecolor='red', facecolor='none', linewidth=2)
        )
        axes[r, c].set_title(f'{row_name} - {col_titles[c]}', pad=3)
        axes[r, c].set_xlabel('Trace')
        axes[r, c].set_ylabel('Sample')
        col_ims[c] = im

# One horizontal colorbar per column at the bottom row
for c, im in enumerate(col_ims):
    if c == 2:
        ticks = [diff_vmin, 0, diff_vmax]
    else:
        ticks = [global_vmin, 0, global_vmax]
    cbar = fig.colorbar(im, cax=axes[3, c], orientation='horizontal', ticks=ticks)
    axes[3, c].set_xlabel('Amplitude', labelpad=3, fontsize=9)
    axes[3, c].tick_params(labelsize=8)

print(f'Dataset: {dataset_name}')
print(f'Color limits (1-99 pct) -> vmin: {global_vmin:.2f}, vmax: {global_vmax:.2f}')
print(f'Diff color limits       -> vmin: {diff_vmin:.2f}, vmax: {diff_vmax:.2f}')
print('Zoom ROI extents (transformed view):')
for row_name, orig, _, _ in rows:
    vis = view_transform(orig)
    h, w = vis.shape
    x0f, x1f, y0f, y1f = zoom_windows[row_name]
    x0 = int(np.clip(round(x0f * w), 0, w - 1))
    x1 = int(np.clip(round(x1f * w), x0 + 1, w))
    y0 = int(np.clip(round(y0f * h), 0, h - 1))
    y1 = int(np.clip(round(y1f * h), y0 + 1, h))
    print(f'  {row_name}: x=[{x0}:{x1}], y=[{y0}:{y1}]')


# In[8]:


# Zoomed 3x3 view using the same ROI windows
rows = [
    ("Inline",    orig_inline,    den_inline,    diff_inline),
    ("Xline",     orig_xline,     den_xline,     diff_xline),
    ("Timeslice", orig_timeslice, den_timeslice, diff_timeslice),
]

col_titles = ['Original (zoom)', 'Denoised (zoom)', 'Removed noise (orig - denoised, zoom)', 'Removed noise (same scale, zoom)']

# Recompute full-view limits here so this cell is self-contained.
orig_den_arrays = [orig_inline, den_inline,
                   orig_xline,  den_xline,
                   orig_timeslice, den_timeslice]
all_vals = np.concatenate([a.ravel() for a in orig_den_arrays])
pct_abs = max(abs(float(np.percentile(all_vals, 1))),
              abs(float(np.percentile(all_vals, 99))))
global_vmin, global_vmax = -pct_abs, pct_abs

diff_arrays = [diff_inline, diff_xline, diff_timeslice]
all_diffs   = np.concatenate([a.ravel() for a in diff_arrays])
diff_abs    = max(abs(float(np.percentile(all_diffs, 1))),
                  abs(float(np.percentile(all_diffs, 99))))
diff_vmin, diff_vmax = -diff_abs, diff_abs

# Build crops from ROI defined in zoom_windows
zoom_data = {}
for row_name, orig, den, diff in rows:
    vis_orig = view_transform(orig)
    vis_den = view_transform(den)
    vis_diff = view_transform(diff)

    h, w = vis_orig.shape
    x0f, x1f, y0f, y1f = zoom_windows[row_name]
    x0 = int(np.clip(round(x0f * w), 0, w - 1))
    x1 = int(np.clip(round(x1f * w), x0 + 1, w))
    y0 = int(np.clip(round(y0f * h), 0, h - 1))
    y1 = int(np.clip(round(y1f * h), y0 + 1, h))

    zoom_data[row_name] = {
        'orig': vis_orig[y0:y1, x0:x1],
        'den': vis_den[y0:y1, x0:x1],
        'diff': vis_diff[y0:y1, x0:x1],
        'extent': (x0, x1, y0, y1),
    }

# Keep 4x4 layout with a dedicated bottom colorbar row
fig_zoom, axes_zoom = plt.subplots(
    4, 4, figsize=(22, 10),
    gridspec_kw={'height_ratios': [10, 10, 10, 0.6], 'hspace': 0.2, 'wspace': 0.25}
)
fig_zoom.suptitle(
    f"Dataset: {dataset_name} | Zoomed ROI from red rectangles",
    fontsize=14, y=0.95
)

# Share X and Y axes row-wise for the image rows
for r in range(3):
    for c in range(1, 4):
        axes_zoom[r, c].sharex(axes_zoom[r, 0])
        axes_zoom[r, c].sharey(axes_zoom[r, 0])

# Plot zoomed crops
col_ims = [None, None, None, None]
for r, (row_name, _, _, _) in enumerate(rows):
    imgs = [
        zoom_data[row_name]['orig'],
        zoom_data[row_name]['den'],
        zoom_data[row_name]['diff'],
        zoom_data[row_name]['diff'],
    ]
    vmins  = [global_vmin, global_vmin, diff_vmin, global_vmin]
    vmaxes = [global_vmax, global_vmax, diff_vmax, global_vmax]

    for c, img in enumerate(imgs):
        im = axes_zoom[r, c].imshow(img, cmap='gray', aspect='auto',
                                    vmin=vmins[c], vmax=vmaxes[c])

        # Titles only on top row
        if r == 0:
            axes_zoom[r, c].set_title(col_titles[c], pad=3)

        # Shared labeling style: y labels on first column only, x labels on last image row only
        if c == 0:
            axes_zoom[r, c].set_ylabel(row_name)
        if r == 2:
            axes_zoom[r, c].set_xlabel('Trace')

        col_ims[c] = im

# One horizontal colorbar per column at the bottom row
for c, im in enumerate(col_ims):
    if c == 2:
        ticks = [diff_vmin, 0, diff_vmax]
    else:
        ticks = [global_vmin, 0, global_vmax]
    cbar = fig_zoom.colorbar(im, cax=axes_zoom[3, c], orientation='horizontal', ticks=ticks)
    cbar.ax.set_xlabel('Amplitude', labelpad=3, fontsize=9)
    cbar.ax.tick_params(labelsize=8)

plt.show()

print('Zoomed ROI extents used (transformed view):')
for row_name in zoom_data:
    x0, x1, y0, y1 = zoom_data[row_name]['extent']
    print(f'  {row_name}: x=[{x0}:{x1}], y=[{y0}:{y1}]')
print(f'Global scale: vmin={global_vmin:.2f}, vmax={global_vmax:.2f}')
print(f'Diff scale: vmin={diff_vmin:.2f}, vmax={diff_vmax:.2f}')


# ## C) Denoise full 3D volume and (optionally) save
# 
# Set `save_output = True` to write the denoised volume to disk. The output path defaults to the same `datasets/` folder; change `output_dir` or `output_filename` as needed.
# 
# The volume is denoised **inline by inline** (each 2-D slice fed through `denoise_section_2d`), so memory usage stays bounded regardless of volume size.
# 

# In[9]:


# ── Output settings ───────────────────────────────────────────────────────────
save_output     = False   # Set to True to write the denoised volume to disk

output_dir      = Path('datasets')          # default: same folder as the input
output_filename = None                      # None → auto: "<stem>_denoised.npy"
                                            # Example override: 'my_denoised.npy'

# ── Full-volume denoising (inline by inline) ──────────────────────────────────
print(f'Denoising full volume: {volume.shape}  (this may take a while on CPU)')

denoised_volume = np.empty_like(volume)
for i in tqdm(range(volume.shape[0]), desc='Inlines'):
    denoised_volume[i] = denoise_section_2d(
        volume[i],
        model,
        device=device,
        batch_size=64,
        stpsz=(16, 16),
        denoise_strength=denoise_strength,
    )

print(f'Denoised volume shape: {denoised_volume.shape}')
print(f'Amplitude range: [{denoised_volume.min():.2f}, {denoised_volume.max():.2f}]')

# ── Save ──────────────────────────────────────────────────────────────────────
if save_output:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if output_filename is None:
        stem = Path(input_path).stem
        output_filename = f'{stem}_denoised.npy'

    out_path = output_dir / output_filename
    np.save(str(out_path), denoised_volume)
    print(f'Saved denoised volume → {out_path.resolve()}')
else:
    print('save_output=False — volume not saved. Set save_output=True to write to disk.')


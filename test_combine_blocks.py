import numpy as np
import time
from sporco import array

# Set up test data similar to the notebook
padded_shape = (1024, 512)
blksz = (128, 128)
stpsz = (16, 16)

# Generate dummy block predictions
# Number of blocks in each dimension:
numblocks = tuple(int(np.floor((a-b)/c) + 1) for a, b, c in zip(padded_shape, blksz, stpsz))
print("Numblocks:", numblocks) # Should be (57, 25) for (1024, 512)

# Total blocks = 57 * 25 = 1425
n_blocks = np.prod(numblocks)
# Blocks flat shape in sporco: blksz + (n_blocks,)
blocks_flat = np.random.randn(*blksz, n_blocks).astype(np.float32)

# 1. Benchmark Sporco combine_blocks
print("Running Sporco combine_blocks...")
start = time.time()
rec_sporco = array.combine_blocks(blocks_flat, padded_shape, stpsz, np.median)
sporco_time = time.time() - start
print(f"Sporco time: {sporco_time:.4f} seconds")

# 2. Vectorized combine_blocks
def combine_blocks_vectorized(blks, imgsz, stpsz, fn=np.median):
    blksz = blks.shape[:-1]
    numblocks = tuple(int(np.floor((a-b)/c) + 1) for a, b, c in zip(imgsz, blksz, stpsz))
    
    # Reshape blks to blksz + numblocks
    new_shape = blksz + numblocks
    blks_reshaped = np.reshape(blks, new_shape)
    
    K_y = int(np.ceil(blksz[0] / stpsz[0]))
    K_x = int(np.ceil(blksz[1] / stpsz[1]))
    max_overlaps = K_y * K_x
    
    if fn == np.mean:
        # Super fast path for mean (no nan, no layers allocation!)
        acc = np.zeros(imgsz, dtype=blks.dtype)
        cnt = np.zeros(imgsz, dtype=blks.dtype)
        for pos in np.ndindex(numblocks):
            i, j = pos
            slices = (slice(i * stpsz[0], i * stpsz[0] + blksz[0]),
                      slice(j * stpsz[1], j * stpsz[1] + blksz[1]))
            acc[slices] += blks_reshaped[(Ellipsis,) + pos]
            cnt[slices] += 1
        return acc / cnt

    layers = np.full((max_overlaps,) + imgsz, np.nan, dtype=blks.dtype)
    
    for pos in np.ndindex(numblocks):
        i, j = pos
        k = (i % K_y) * K_x + (j % K_x)
        slices = (slice(i * stpsz[0], i * stpsz[0] + blksz[0]),
                  slice(j * stpsz[1], j * stpsz[1] + blksz[1]))
        layers[k][slices] = blks_reshaped[(Ellipsis,) + pos]
        
    if fn == np.median:
        return np.nanmedian(layers, axis=0)
    else:
        # Fallback
        return np.nanapply_along_axis(fn, 0, layers)

print("Running Vectorized combine_blocks with np.median...")
start = time.time()
rec_vectorized_med = combine_blocks_vectorized(blocks_flat, padded_shape, stpsz, np.median)
vectorized_time_med = time.time() - start
print(f"Vectorized median time: {vectorized_time_med:.4f} seconds")

print("Running Vectorized combine_blocks with np.mean...")
start = time.time()
rec_vectorized_mean = combine_blocks_vectorized(blocks_flat, padded_shape, stpsz, np.mean)
vectorized_time_mean = time.time() - start
print(f"Vectorized mean time: {vectorized_time_mean:.4f} seconds")

# Benchmark Sporco combine_blocks with mean
print("Running Sporco combine_blocks with np.mean...")
start = time.time()
rec_sporco_mean = array.combine_blocks(blocks_flat, padded_shape, stpsz, np.mean)
sporco_time_mean = time.time() - start
print(f"Sporco mean time: {sporco_time_mean:.4f} seconds")

# Verify correctness
print("Median match:", np.allclose(rec_sporco, rec_vectorized_med, equal_nan=True))
print("Mean match:", np.allclose(rec_sporco_mean, rec_vectorized_mean, equal_nan=True))
print(f"Mean Speedup: {sporco_time_mean / vectorized_time_mean:.1f}x")


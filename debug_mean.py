import numpy as np
from test_combine_blocks import combine_blocks_vectorized, blocks_flat, padded_shape, stpsz
from sporco import array

rec_vectorized_mean = combine_blocks_vectorized(blocks_flat, padded_shape, stpsz, np.mean)
rec_sporco_mean = array.combine_blocks(blocks_flat, padded_shape, stpsz, np.mean)

# Print differences
diff = np.abs(rec_sporco_mean - rec_vectorized_mean)
print("Max diff:", np.max(np.nan_to_num(diff)))
print("Nans in sporco:", np.isnan(rec_sporco_mean).sum())
print("Nans in vectorized:", np.isnan(rec_vectorized_mean).sum())

# Find a pixel where they differ
idx = np.where(diff > 1e-5)
if len(idx[0]) > 0:
    y, x = idx[0][0], idx[1][0]
    print(f"Difference at ({y}, {x}): sporco={rec_sporco_mean[y, x]}, vectorized={rec_vectorized_mean[y, x]}")

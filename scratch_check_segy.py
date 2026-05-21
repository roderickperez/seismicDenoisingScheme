import segyio
import numpy as np
from pathlib import Path

p = Path('datasets/1_Original_Seismics.sgy')

try:
    with segyio.open(str(p), strict=False, ignore_geometry=True) as f:
        traces = segyio.tools.collect(f.trace[:]).astype(np.float32)  # [n_traces, n_samples]
        print("Traces collected shape:", traces.shape)
        
        il = np.array(f.attributes(segyio.TraceField.INLINE_3D)[:])
        xl = np.array(f.attributes(segyio.TraceField.CROSSLINE_3D)[:])
        
        uil = np.unique(il)
        uxl = np.unique(xl)
        print("Unique Inlines:", len(uil), "min:", uil.min(), "max:", uil.max())
        print("Unique Crosslines:", len(uxl), "min:", uxl.min(), "max:", uxl.max())
        
        # Calculate size
        expected_size = len(uil) * len(uxl) * traces.shape[1]
        print("Expected array elements:", expected_size)
        print("Expected size in GB:", expected_size * 4 / (1024**3))
        
except Exception as e:
    print("Error:", str(e))

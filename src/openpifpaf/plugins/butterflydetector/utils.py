import functools

import numpy as np

@functools.lru_cache(maxsize=64)
def create_sink_2d(w, h):
    if w == 1 and h == 1:
        return np.zeros((2, 1, 1))

    sink1d_w = np.linspace((w - 1.0) / 2.0, -(w - 1.0) / 2.0, num=w, dtype=np.float32)
    sink1d_h = np.linspace((h - 1.0) / 2.0, -(h - 1.0) / 2.0, num=h, dtype=np.float32)
    sink = np.stack((
        sink1d_w.reshape(1, -1).repeat(h, axis=0),
        sink1d_h.reshape(-1, 1).repeat(w, axis=1),
    ), axis=0)
    return sink

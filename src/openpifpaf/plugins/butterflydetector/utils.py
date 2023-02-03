import functools

import numpy as np

@functools.lru_cache(maxsize=16)
def index_field(shape):
    yx = np.indices(shape, dtype=np.float32)
    xy = np.flip(yx, axis=0)
    return xy
    
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

def normalize_butterfly(joint_intensity_fields, joint_fields, joint_fields_b, width_fields, height_fields, *,
                  fixed_scale=None):
    joint_intensity_fields = np.expand_dims(joint_intensity_fields.copy(), 1)
    width_fields = np.expand_dims(width_fields, 1)
    height_fields = np.expand_dims(height_fields, 1)
    if fixed_scale is not None:
        width_fields[:] = width_fields
        height_fields[:] = height_fields

    index_fields = index_field(joint_fields.shape[-2:])
    index_fields = np.expand_dims(index_fields, 0)
    joint_fields = index_fields + joint_fields

    return np.concatenate(
        (joint_intensity_fields, joint_fields, width_fields, height_fields),
        axis=1,
    ), joint_fields_b

def scalar_square_add_2dsingle(field, x, y, width, height, value):
    minx = max(0, int(x - width))
    miny = max(0, int(y - height))
    maxx = max(minx + 1, min(field.shape[1], int(x + width) + 1))
    maxy = max(miny + 1, min(field.shape[0], int(y + height) + 1))
    field[miny:maxy, minx:maxx] += value

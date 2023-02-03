# cython: infer_types=True
cimport cython
from libc.math cimport exp, fabs, sqrt, fmin, fmax
import numpy as np


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef unsigned char scalar_nonzero(unsigned char[:, :] field, float x, float y, unsigned char default=0):
    if x < 0.0 or y < 0.0 or x > field.shape[1] - 1 or y > field.shape[0] - 1:
        return default

    return field[<Py_ssize_t>y, <Py_ssize_t>x]

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef void scalar_square_add_2dgauss(float[:, :] field, float[:] x, float[:] y, float[:] sigma_w, float[:] sigma_h, float[:] v, float truncate=2.0) nogil:
    cdef Py_ssize_t i, xx, yy
    cdef float vv, deltax2, deltay2
    cdef float cv, cx, cy, csigma_w, csigma_w2, csigma_h, csigma_h2
    cdef long minx, miny, maxx, maxy

    for i in range(x.shape[0]):
        csigma_w = sigma_w[i]
        csigma_w2 = csigma_w * csigma_w
        csigma_h = sigma_h[i]
        csigma_h2 = csigma_h * csigma_h
        cx = x[i]
        cy = y[i]
        cv = v[i]

        minx = (<long>clip(cx - truncate * csigma_w, 0, field.shape[1] - 1))
        maxx = (<long>clip(cx + truncate * csigma_w, minx + 1, field.shape[1]))
        miny = (<long>clip(cy - truncate * csigma_h, 0, field.shape[0] - 1))
        maxy = (<long>clip(cy + truncate * csigma_h, miny + 1, field.shape[0]))
        for xx in range(minx, maxx):
            deltax2 = (xx - cx)**2
            for yy in range(miny, maxy):
                deltay2 = (yy - cy)**2
                vv = cv * approx_exp(-0.5 * (deltax2/csigma_w2 + deltay2/csigma_h2))
                field[yy, xx] += vv

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef void cumulative_average_2d(float[:, :] cuma, float[:, :] cumw, float[:] x, float[:] y, float[:] width, float[:] height, float[:] v, float[:] w) nogil:
    cdef long minx, miny, maxx, maxy
    cdef float cv, cw, cx, cy, cwidth, cheight
    cdef Py_ssize_t i, xx, yy

    for i in range(x.shape[0]):
        cw = w[i]
        if cw <= 0.0:
            continue

        cv = v[i]
        cx = x[i]
        cy = y[i]
        cwidth = width[i]
        cheight = height[i]

        minx = (<long>clip(cx - cwidth, 0, cuma.shape[1] - 1))
        maxx = (<long>clip(cx + cwidth, minx + 1, cuma.shape[1]))
        miny = (<long>clip(cy - cheight, 0, cuma.shape[0] - 1))
        maxy = (<long>clip(cy + cheight, miny + 1, cuma.shape[0]))
        for xx in range(minx, maxx):
            for yy in range(miny, maxy):
                cuma[yy, xx] = (cw * cv + cumw[yy, xx] * cuma[yy, xx]) / (cumw[yy, xx] + cw)
                cumw[yy, xx] += cw

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef void scalar_square_add_2dgauss_with_max(float[:, :] field, float[:] x, float[:] y, float[:] sigma_w, float[:] sigma_h, float[:] v, float truncate=2.0, float max_value=1.0) nogil:
    cdef Py_ssize_t i, xx, yy
    cdef float vv, deltax2, deltay2
    cdef float cv, cx, cy, csigma_w, csigma_w2, csigma_h, csigma_h2
    cdef long minx, miny, maxx, maxy

    for i in range(x.shape[0]):
        csigma_w = sigma_w[i]
        csigma_w2 = csigma_w * csigma_w
        csigma_h = sigma_h[i]
        csigma_h2 = csigma_h * csigma_h
        cx = x[i]
        cy = y[i]
        cv = v[i]

        minx = (<long>clip(cx - truncate * csigma_w, 0, field.shape[1] - 1))
        maxx = (<long>clip(cx + truncate * csigma_w, minx + 1, field.shape[1]))
        miny = (<long>clip(cy - truncate * csigma_h, 0, field.shape[0] - 1))
        maxy = (<long>clip(cy + truncate * csigma_h, miny + 1, field.shape[0]))
        for xx in range(minx, maxx):
            deltax2 = (xx - cx)**2
            for yy in range(miny, maxy):
                deltay2 = (yy - cy)**2
                vv = cv * approx_exp(-0.5 * (deltax2/csigma_w2 + deltay2/csigma_h2))
                field[yy, xx] += vv
                field[yy, xx] = min(max_value, field[yy, xx])

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef void scalar_square_average_2dgauss_with_max(float[:, :] field, float[:] x, float[:] y, float[:] sigma_w, float[:] sigma_h, float[:] v, float[:,:] cumN, float truncate=2.0, float max_value=1.0) nogil:
    cdef Py_ssize_t i, xx, yy
    cdef float vv, deltax2, deltay2
    cdef float cv, cx, cy, csigma_w, csigma_w2, csigma_h, csigma_h2
    cdef long minx, miny, maxx, maxy

    for i in range(x.shape[0]):
        csigma_w = sigma_w[i]
        csigma_w2 = csigma_w * csigma_w
        csigma_h = sigma_h[i]
        csigma_h2 = csigma_h * csigma_h
        cx = x[i]
        cy = y[i]
        cv = v[i]

        minx = (<long>clip(cx - truncate * csigma_w, 0, field.shape[1] - 1))
        maxx = (<long>clip(cx + truncate * csigma_w, minx + 1, field.shape[1]))
        miny = (<long>clip(cy - truncate * csigma_h, 0, field.shape[0] - 1))
        maxy = (<long>clip(cy + truncate * csigma_h, miny + 1, field.shape[0]))
        for xx in range(minx, maxx):
            deltax2 = (xx - cx)**2
            for yy in range(miny, maxy):
                deltay2 = (yy - cy)**2
                vv = cv * approx_exp(-0.5 * (deltax2/csigma_w2 + deltay2/csigma_h2))
                field[yy, xx] = (vv+field[yy, xx]*cumN[yy, xx])/ (cumN[yy, xx] + cv)
                field[yy, xx] = min(max_value, field[yy, xx])
                cumN[yy, xx] += cv

@cython.boundscheck(False)
@cython.wraparound(False)
def scalar_values(float[:, :] field, float[:] x, float[:] y, float default=-1):
    values_np = np.full((x.shape[0],), default, dtype=np.float32)
    cdef float[:] values = values_np
    cdef float maxx = <float>field.shape[1] - 1, maxy = <float>field.shape[0] - 1

    for i in range(values.shape[0]):
        if x[i] < 0.0 or y[i] < 0.0 or x[i] > maxx or y[i] > maxy:
            continue

        values[i] = field[<Py_ssize_t>y[i], <Py_ssize_t>x[i]]

    return values_np

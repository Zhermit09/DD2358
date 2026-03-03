# cython: boundscheck=False, wraparound=False, cdivision=True
import numpy as np

def gauss_seidel_cython(double[:, :] f):
    cdef int N = f.shape[0]
    cdef int i, j
    for i in range(1, N - 1):
        for j in range(1, N - 1):
            f[i, j] = 0.25 * (f[i, j+1] + f[i, j-1] +
                               f[i+1, j] + f[i-1, j])
    return f

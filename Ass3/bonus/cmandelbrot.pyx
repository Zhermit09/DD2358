cimport cython
import numpy as np
# noinspection PyUnresolvedReferences
cimport numpy as np

np.import_array()

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double[:, :] mandelbrot_set(unsigned int width, unsigned int height, double x_min, double x_max,
                                  double y_min, double y_max, unsigned int max_iter=100):
    """Generates the Mandelbrot set image."""
    cdef double[:] x_vals = np.linspace(x_min, x_max, width, dtype=np.double)
    cdef double[:] y_vals = np.linspace(y_min, y_max, height, dtype=np.double)
    cdef double[:, :] image = np.empty((height, width), dtype=np.double)
    cdef unsigned int i, j
    cdef double complex c

    for i in range(height):
        for j in range(width):
            c = x_vals[j] + y_vals[i] * 1j
            image[i, j] = mandelbrot(c, max_iter)

    return image

cdef inline unsigned int mandelbrot(double complex c, unsigned int max_iter=100) noexcept:
    """Computes the number of iterations before divergence."""
    cdef double complex z = 0
    cdef unsigned int n
    for n in range(max_iter):
        if abs(z) > 2:
            return n
        z = z * z + c
    return max_iter

import numpy as np
import matplotlib.pyplot as plt


def mandelbrot(C, max_iter=100):
    """Computes the number of iterations before divergence."""
    image = np.full(C.shape, max_iter, dtype=int)
    Z = np.zeros(C.shape, dtype=complex)
    mask = np.ones(C.shape, dtype=bool)
    diverged = np.zeros(C.shape, dtype=bool)

    for n in range(max_iter):
        diverged[mask] = np.abs(Z[mask]) > 2
        image[diverged] = n
        mask[diverged] = False
        diverged.fill(False)

        Z[mask] = Z[mask] * Z[mask] + C[mask]

        if not mask.any(): break
    return image


def mandelbrot_set(width, height, x_min, x_max, y_min, y_max, max_iter=100):
    """Generates the Mandelbrot set image."""
    x_vals = np.linspace(x_min, x_max, width)
    y_vals = np.linspace(y_min, y_max, height)

    real, imag = np.meshgrid(x_vals, y_vals, indexing="xy")
    C = real + imag * 1j

    return mandelbrot(C, max_iter)


# Parameters
width, height = 1000, 800
x_min, x_max, y_min, y_max = -2, 1, -1, 1

# Generate fractal
image = mandelbrot_set(width, height, x_min, x_max, y_min, y_max)

# Display
plt.imshow(image, cmap='inferno', extent=[x_min, x_max, y_min, y_max])
plt.colorbar()
plt.title("Mandelbrot Set")
plt.show()

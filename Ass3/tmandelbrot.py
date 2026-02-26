import torch as th
import matplotlib.pyplot as plt

def tmandelbrot(C, max_iter=100):
    """Computes the number of iterations before divergence."""
    device = C.device

    image = th.full(C.shape, max_iter, dtype=th.int32, device=device)
    Z = th.zeros(C.shape, dtype=th.complex128, device=device)
    mask = th.ones(C.shape, dtype=th.bool, device=device)
    diverged = th.zeros(C.shape, dtype=th.bool, device=device)

    for n in range(max_iter):
        diverged[mask] = th.abs(Z[mask]) > 2
        image[diverged] = n
        mask[diverged] = False
        diverged.fill_(False)

        Z[mask] = Z[mask] * Z[mask] + C[mask]

        if not mask.any(): break
    return image


def tmandelbrot_set(width, height, x_min, x_max, y_min, y_max, max_iter=100):
    """Generates the Mandelbrot set image."""
    device = th.device("cuda")
    x_vals = th.linspace(x_min, x_max, width, dtype=th.float64, device=device)
    y_vals = th.linspace(y_min, y_max, height, dtype=th.float64, device=device)

    real, imag = th.meshgrid(x_vals, y_vals, indexing="xy")
    C = th.complex(real, imag)

    return tmandelbrot(C, max_iter)


# Parameters
width, height = 1000, 800
x_min, x_max, y_min, y_max = -2, 1, -1, 1

# Generate fractal
image = tmandelbrot_set(width, height, x_min, x_max, y_min, y_max).cpu().numpy()

# Display
plt.imshow(image, cmap='inferno', extent=[x_min, x_max, y_min, y_max])
plt.colorbar()
plt.title("Mandelbrot Set")
plt.show()

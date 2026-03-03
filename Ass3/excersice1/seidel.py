from array import array
import random
import time

import numpy as np

# ---------------------------------------------------------------------------
# Task 1.1 – Gauss-Seidel solver: Python list, array, NumPy
# Benchmark all three implementations across varying grid sizes.
# ---------------------------------------------------------------------------

ITERS      = 100
GRID_SIZES = [16, 32, 64, 128, 256]
SEED       = 42

# ── Helpers ─────────────────────────────────────────────────────────────────

def make_grid_numpy(N):
    rng = np.random.default_rng(SEED)
    f = rng.random((N, N))
    f[0, :] = f[-1, :] = f[:, 0] = f[:, -1] = 0.0
    return f

def make_grid_list(N):
    random.seed(SEED)
    f = [[0.0] * N for _ in range(N)]
    for i in range(1, N - 1):
        for j in range(1, N - 1):
            f[i][j] = random.random()
    return f

def make_grid_array(N):
    """Flat Python array('d') of length N*N: random interior, zero boundary."""
    f = make_grid_list(N)
    return array('d', (f[i][j] for i in range(N) for j in range(N))), N

# ── Implementations ─────────────────────────────────────────────────────────

def gauss_seidel_numpy(f):
    """Gauss-Seidel sweep on a NumPy 2-D array (true GS ordering)."""
    for i in range(1, f.shape[0] - 1):
        for j in range(1, f.shape[1] - 1):
            f[i, j] = 0.25 * (f[i, j+1] + f[i, j-1] +
                               f[i+1, j] + f[i-1, j])
    return f

def gauss_seidel_list(f):
    """Gauss-Seidel sweep on a Python list-of-lists grid."""
    N = len(f)
    for i in range(1, N - 1):
        for j in range(1, N - 1):
            f[i][j] = 0.25 * (f[i][j+1] + f[i][j-1] +
                               f[i+1][j] + f[i-1][j])
    return f

def gauss_seidel_array(f, N):
    for i in range(1, N - 1):
        for j in range(1, N - 1):
            idx = i * N + j
            f[idx] = 0.25 * (f[idx + 1]     +  
                              f[idx - 1]     + 
                              f[idx + N]     +  
                              f[idx - N])       
    return f

# ── Benchmark ────────────────────────────────────────────────────────────────

def benchmark():
    times_numpy = []
    times_list  = []
    times_array = []

    for N in GRID_SIZES:
        print(f"N = {N} ...", end=" ", flush=True)

        # --- NumPy ---
        f = make_grid_numpy(N)
        t0 = time.perf_counter()
        for _ in range(ITERS):
            gauss_seidel_numpy(f)
        times_numpy.append(time.perf_counter() - t0)

        # --- list ---
        f = make_grid_list(N)
        t0 = time.perf_counter()
        for _ in range(ITERS):
            gauss_seidel_list(f)
        times_list.append(time.perf_counter() - t0)

        # --- array ---
        f, _ = make_grid_array(N)
        t0 = time.perf_counter()
        for _ in range(ITERS):
            gauss_seidel_array(f, N)
        times_array.append(time.perf_counter() - t0)

        print(f"list={times_list[-1]:.3f}s  array={times_array[-1]:.3f}s  numpy={times_numpy[-1]:.3f}s")

    return times_list, times_array, times_numpy


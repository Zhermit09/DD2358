"""
benchmark_grid.py — serial vs parallel simulation at increasing grid sizes.

Grid nr controls domain size: nz = nr*3, so nr=24 is 4x more cells than nr=12.
Larger grids also produce more particles, increasing the parallel push workload.

Run: python Project/benchmark_grid.py
"""

import sys
import time

sys.path.insert(0, "Project")
sys.path.insert(0, "Project/Cpu")

import rz_pic as serial
import rz_pic_cpu as parallel

grid_sizes = [12, 18, 24, 36, 48]

if __name__ == "__main__":
    print(f"{'NR':>6}  {'NZ':>6}  {'Serial (s)':>12}  {'Parallel (s)':>14}  {'Speedup':>9}")
    print("-" * 55)

    for nr in grid_sizes:
        # --- serial ---
        serial.reassign_globals(nr)
        t0 = time.perf_counter()
        serial.main()
        t_serial = time.perf_counter() - t0

        # --- parallel ---
        parallel.reassign_globals(nr)
        t0 = time.perf_counter()
        parallel.main()
        t_parallel = time.perf_counter() - t0

        print(f"{nr:>6}  {nr*3:>6}  {t_serial:>12.2f}  {t_parallel:>14.2f}  {t_serial/t_parallel:>8.2f}x")

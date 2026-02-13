import conway2
import timeit
import matplotlib.pyplot as plt

grid_sizes = [30, 60, 90, 120, 150, 180, 210, 240, 270, 300]

time = []
for size in grid_sizes:
    t1 = timeit.default_timer()
    conway2.main(size)
    t2 = timeit.default_timer()
    time.append((t2 - t1))

plt.figure(figsize=(8, 5))
plt.plot(grid_sizes, time, marker='o')
plt.xlabel("Grid size")
plt.ylabel("Execution time (s)")
plt.title(f"Benchmark: conway2.py (100 iterations)")
plt.grid(True)
plt.savefig("benchmark2.png", dpi=300)
plt.show()
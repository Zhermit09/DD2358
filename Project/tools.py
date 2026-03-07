import json
import random
from random import seed

import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from datetime import datetime
from timeit import default_timer as timer


def validate(base, changed, nr=12, seed=42):
    base.reassign_globals(nr)
    random.seed(seed)
    base.main()
    den1 = base.den
    phi1 = base.phi

    changed.reassign_globals(nr)
    random.seed(seed)
    changed.main()
    den2 = changed.den
    phi2 = changed.phi

    np.testing.assert_equal(den1, den2)
    np.testing.assert_equal(phi1, phi2)
    print("Valid for nr:", nr)


def benchmark_run(module, nr, args=(), n=10):
    print(f"[{datetime.now()}]", f"Benchmarking ({module.__name__}):\t", nr)
    times = []
    for _ in range(n):
        module.reassign_globals(nr)

        start = timer()
        module.main(*args)
        end = timer()

        times.append(end - start)

    return np.mean(times), np.std(times, ddof=1)


def plot(data):
    plt.figure(figsize=(9, 5))

    for nrs, avgs, stds, name in data:
        nrs = np.array(nrs)
        avgs = np.array(avgs)
        stds = np.array(stds)

        line, = plt.plot(nrs, avgs, marker='o', label=name)
        plt.fill_between(nrs, avgs - stds, avgs + stds, color=line.get_color(), alpha=0.4)

    plt.xlabel("Grid height (nr)")
    plt.ylabel("Execution time (s)")
    plt.legend()
    plt.grid(True)
    plt.show()


def benchmark(modules, nrss, args, names, n=10, save=True):
    data = []
    for module, nrs, arg, name in zip(modules, nrss, args, names):
        avgs, stds = [], []
        for nr in nrs:
            avg, std = benchmark_run(module, nr, arg, n)
            avgs.append(avg)
            stds.append(std)
        print(f"[{datetime.now()}]", f"{module.__name__} is done!\n")
        data.append((nrs, avgs, stds, name))

    if save:
        path = Path("benchmark/data.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as file:
            json.dump({"_".join(names): data}, file, indent=3)
        print("Data Saved\n")

    return data


import rz_pic
import C.rz_pic_C as rz_pic_C

# seed(42)
# rz_pic.main()
#validate(rz_pic_C, rz_pic, 8)
#validate(rz_pic_C, rz_pic, 10)

nrs = [8, 12, 14]
data = benchmark([rz_pic_C, rz_pic], [nrs, nrs], [(), ()], ["cython", "base"], 3)
plot(data)

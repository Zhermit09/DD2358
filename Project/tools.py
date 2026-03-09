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

    np.testing.assert_allclose(den1, den2)
    np.testing.assert_allclose(phi1, phi2)
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
    path = Path("benchmark")
    path.parent.mkdir(parents=True, exist_ok=True)
    names = []

    plt.figure(figsize=(9, 5))

    for nrs, avgs, stds, name in data:
        names.append(name)

        nrs = np.array(nrs)
        avgs = np.array(avgs)
        stds = np.array(stds)

        line, = plt.plot(nrs, avgs, marker='o', label=name)
        plt.fill_between(nrs, avgs - stds, avgs + stds, color=line.get_color(), alpha=0.4)

    plt.xlabel("Grid height (nr)")
    plt.ylabel("Execution time (s)")
    plt.legend()
    plt.grid(True)
    plt.savefig(path / f"{"_".join(names)}.pdf", bbox_inches="tight")
    plt.show()


def benchmark(modules, nrss, args, names, n=10, save=True):
    seed()
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
        JSON = {}
        try:
            with open(path, "r") as f:
                JSON = json.load(f)
        except Exception:
            pass

        with open(path, "w") as file:
            JSON["_".join(names)] = data
            json.dump(JSON, file, indent=3)
        print("Data Saved\n")

    return data


import rz_pic
import C.rz_pic_C as rz_pic_C
import GPU.rz_pic_GPU as rz_pic_GPU

nrs = [8, 10, 12, 14, 18, 22, 26, 30]
for nr in nrs: validate(rz_pic, rz_pic_C, nr)
data = benchmark([rz_pic, rz_pic_C], [nrs, nrs], [(), ()], ["Python", "Cython"])
plot(data)

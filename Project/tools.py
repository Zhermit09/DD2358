import json
import random

import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from datetime import datetime
from timeit import default_timer as timer


def reassign_globals(module, nr=12):
    def XtoL(pos):
        lc = [pos[0] / dz, pos[1] / dr]
        return lc

    nz = nr * 3
    dz = 1e-3
    dr = 1e-3
    dt = 5e-9

    QE = 1.602e-19
    AMU = 1.661e-27
    EPS0 = 8.854e-12

    charge = QE
    m = 40 * AMU  # argon ions
    qm = charge / m
    spwt = 50

    # solver parameters
    n0 = 1e12
    phi0 = 100
    phi1 = 0
    kTe = 5

    phi = np.zeros([nz, nr])
    efz = np.zeros([nz, nr])
    efr = np.zeros([nz, nr])
    rho_i = np.zeros([nz, nr])
    den = np.zeros([nz, nr])

    # ---- sugarcube domain --------------------
    cell_type = np.zeros([nz, nr])
    tube1_radius = (nr / 2) * dr
    tube1_length = 0.28 * nz * dz
    tube1_aperture_rad = (nr / 3) * dr
    tube2_radius = tube1_radius + dr
    tube2_length = tube1_length + 2 * dz
    tube2_aperture_rad = (nr / 4) * dr
    [tube_i_max, tube_j_max] = map(int, XtoL([4 * dz, tube1_radius]))

    globals_dict = {
        "nr": nr,
        "nz": nz,
        "dz": dz,
        "dr": dr,
        "dt": dt,
        "QE": QE,
        "AMU": AMU,
        "EPS0": EPS0,
        "charge": charge,
        "m": m,
        "qm": qm,
        "spwt": spwt,
        "n0": n0,
        "phi0": phi0,
        "phi1": phi1,
        "kTe": kTe,
        "phi": phi,
        "efz": efz,
        "efr": efr,
        "rho_i": rho_i,
        "den": den,
        "cell_type": cell_type,
        "tube1_radius": tube1_radius,
        "tube1_length": tube1_length,
        "tube1_aperture_rad": tube1_aperture_rad,
        "tube2_radius": tube2_radius,
        "tube2_length": tube2_length,
        "tube2_aperture_rad": tube2_aperture_rad,
        "tube_i_max": tube_i_max,
        "tube_j_max": tube_j_max,
    }

    for name, val in globals_dict.items():
        setattr(module, name, val)


def validate(base, changed, nr=12, seed=42):
    reassign_globals(base, nr)
    random.seed(seed)
    base.main()
    den1 = base.den
    phi1 = base.phi

    reassign_globals(changed, nr)
    random.seed(seed)
    changed.main()
    den2 = changed.den
    phi2 = changed.phi

    np.testing.assert_allclose(den1, den2)
    np.testing.assert_allclose(phi1, phi2)


def benchmark_run(module, nr, args=(), n=10):
    print(f"[{datetime.now()}]", f"Benchmarking ({module.__name__}):\t", nr)
    times = []
    for _ in range(n):
        reassign_globals(module, nr)

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


"""import rz_pic

nrs = [8, 12, 14]
data = benchmark([rz_pic, rz_pic], [nrs, nrs], [(), ()], ["test", "test2"])
plot(data)"""

import subprocess
import time
import sys
import argparse

import psutil
import matplotlib.pyplot as plt


def start():
    return time.perf_counter()

def end():
    return time.perf_counter()


def print_summary(samples):
    if not samples:
        print("No CPU samples collected.")
        return

    ncores = len(samples[0])

    print("\nSummary (per core):")
    print(f"{'Core':>6} {'Mean%':>8} {'Max%':>8}")
    for c in range(ncores):
        col = [row[c] for row in samples]
        mean = sum(col) / len(col)
        mx = max(col)
        print(f"{c:>6} {mean:>8.2f} {mx:>8.2f}")


def plot_cpu(times, samples, out_png):
    if not samples:
        print("No CPU samples to plot.")
        return

    ncores = len(samples[0])
    for c in range(ncores):
        y = [row[c] for row in samples]
        plt.plot(times, y, label=f"core {c}")

    plt.xlabel("Time (s)")
    plt.ylabel("CPU usage (%)")
    plt.title("CPU usage per core during execution")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"Saved plot: {out_png}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)
    parser.add_argument("--file-args", nargs="*", default=[])
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--output", default="cpu_profile.png")
    args = parser.parse_args()

    t0 = start()

    # CPU sampling storage
    times = []
    samples = []

    # Start the target program (non-blocking)
    p = subprocess.Popen([sys.executable, args.file] + args.file_args)

    psutil.cpu_percent(interval = args.interval, percpu=True)

    # Sample while the process is running
    while p.poll() is None: 
        per_core = psutil.cpu_percent(interval = args.interval, percpu = True)  
        times.append(time.perf_counter() - t0)
        samples.append(per_core)

    # Ensure process finished and get return code
    rc = p.wait()

    t1 = end()
    print(f"\nReturn code: {rc}")
    print(f"Elapsed time: {(t1 - t0):.3f} seconds")

    plot_cpu(times, samples, args.output)
    print_summary(samples)


if __name__ == "__main__":
    main()

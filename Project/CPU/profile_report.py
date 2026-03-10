import sys
import os
import cProfile
import pstats
import io
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))  
sys.path.insert(0, _HERE)                      

import rz_pic as serial_mod
import rz_pic_CPU as parallel_mod

GRID_SIZES = [12, 24, 36]
TOP_N = 8 

SIM_KEYWORDS = ["solvePotential", "gather", "scatter", "XtoL",
                "computeEF", "push_chunk", "main"]


def profile_run(mod, nr):
    mod.reassign_globals(nr)
    pr = cProfile.Profile()
    t0 = time.perf_counter()
    pr.enable()
    mod.main()
    pr.disable()
    elapsed = time.perf_counter() - t0

    ps = pstats.Stats(pr, stream=io.StringIO())
    ps.strip_dirs()
    ps.sort_stats("cumulative")

    capture = io.StringIO()
    ps.stream = capture
    ps.print_stats(TOP_N)
    raw = capture.getvalue()
    top_rows = []
    in_table = False
    for line in raw.splitlines():
        if "ncalls" in line:
            in_table = True
            continue
        if in_table and line.strip():
            parts = line.split()
            if len(parts) >= 6:
                top_rows.append((parts[0], parts[3], parts[4], " ".join(parts[5:])))

    sim_rows = []
    for stat_key, stat_val in ps.stats.items():
        fname = stat_key[2]
        if any(kw in fname for kw in SIM_KEYWORDS):
            cc, nc, tt, ct, _ = stat_val
            sim_rows.append((str(nc), f"{ct:.3f}", f"{ct/nc:.6f}" if nc else "0", fname))
    sim_rows.sort(key=lambda r: float(r[1]), reverse=True)

    return elapsed, top_rows, sim_rows


def print_table(rows):
    print(f"  {'ncalls':>8}  {'cumtime(s)':>11}  {'percall(s)':>11}  function")
    print(f"  {'-'*8}  {'-'*11}  {'-'*11}  {'-'*34}")
    for r in rows:
        print(f"  {r[0]:>8}  {r[1]:>11}  {r[2]:>11}  {r[3]}")


def run():
    print("=" * 72)
    print("  RZ-PIC PROFILING REPORT  —  Serial vs Parallel  (1000 timesteps)")
    print("=" * 72)

    timing_table = []

    for nr in GRID_SIZES:
        nz = nr * 3
        print(f"\n{'─'*72}")
        print(f"  Grid NR={nr}  NZ={nz}  (total nodes = {nr*nz})")
        print(f"{'─'*72}")

        # --- SERIAL ---
        print(f"\n  [SERIAL]  top {TOP_N} by cumulative time")
        t_ser, top_ser, _ = profile_run(serial_mod, nr)
        print(f"  Total wall time: {t_ser:.2f}s")
        print_table(top_ser)

        # --- PARALLEL ---
        print(f"\n  [PARALLEL]")
        t_par, top_par, sim_par = profile_run(parallel_mod, nr)
        speedup = t_ser / t_par if t_par > 0 else float("inf")
        print(f"  Total wall time: {t_par:.2f}s   (speedup: {speedup:.2f}x)")

        print(f"\n  Top {TOP_N} by cumtime (IPC wait dominates because workers")
        print(f"  run in sub-processes invisible to cProfile):")
        print_table(top_par)

        print(f"\n  Simulation functions only (shows bottleneck shift):")
        print_table(sim_par[:8])

        timing_table.append((nr, nz, t_ser, t_par, speedup))

    # --- SUMMARY TABLE ---
    print(f"\n{'='*72}")
    print("  OVERALL TIMING SUMMARY")
    print(f"{'='*72}")
    print(f"  {'NR':>4}  {'NZ':>4}  {'Serial(s)':>10}  {'Parallel(s)':>12}  {'Speedup':>9}")
    print(f"  {'-'*4}  {'-'*4}  {'-'*10}  {'-'*12}  {'-'*9}")
    for nr, nz, ts, tp, sp in timing_table:
        print(f"  {nr:>4}  {nz:>4}  {ts:>10.2f}  {tp:>12.2f}  {sp:>8.2f}x")
    print()
    print("  NOTE: Parallel workers (push_chunk_shm) run in sub-processes.")
    print("  cProfile cannot see inside them. IPC wait in 'connection.*'")
    print("  reflects the main process blocking while workers push particles.")


class _Tee:
    """Write to both the real stdout and a file at the same time."""
    def __init__(self, filepath):
        self._file = open(filepath, "w", encoding="utf-8")
        self._stdout = sys.stdout

    def write(self, data):
        self._stdout.write(data)
        self._file.write(data)

    def flush(self):
        self._stdout.flush()
        self._file.flush()

    def close(self):
        self._file.close()


if __name__ == "__main__":
    _out_path = os.path.join(_HERE, "profile_output.txt")
    _tee = _Tee(_out_path)
    sys.stdout = _tee
    try:
        run()
    finally:
        sys.stdout = _tee._stdout
        _tee.close()
    print(f"\nReport saved to: {_out_path}")

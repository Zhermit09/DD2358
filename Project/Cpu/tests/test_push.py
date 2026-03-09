import sys
import os
import math
import numpy
from multiprocessing import Pool, shared_memory as shm_module

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rz_pic_cpu import push_chunk_shm, N_CORES


DZ = DR = 1e-3
DT = 5e-9
QM = 1.602e-19 / (40 * 1.661e-27)

EFZ = numpy.array([
    [  0.0,    0.0,    0.0,    0.0,    0.0,    0.0],
    [  0.0,  200.0,  400.0,  600.0,  800.0, 1000.0],
    [  0.0,  400.0,  800.0, 1200.0, 1600.0, 2000.0],
    [  0.0,  600.0, 1200.0, 1800.0, 2400.0, 3000.0],
    [  0.0,  800.0, 1600.0, 2400.0, 3200.0, 4000.0],
    [  0.0, 1000.0, 2000.0, 3000.0, 4000.0, 5000.0],
], dtype=float)

EFR = numpy.array([
    [   0.0,    0.0,    0.0,    0.0,    0.0,    0.0],
    [   0.0, -100.0, -200.0, -300.0, -400.0, -500.0],
    [   0.0, -200.0, -400.0, -600.0, -800.0,-1000.0],
    [   0.0, -300.0, -600.0, -900.0,-1200.0,-1500.0],
    [   0.0, -400.0, -800.0,-1200.0,-1600.0,-2000.0],
    [   0.0, -500.0,-1000.0,-1500.0,-2000.0,-2500.0],
], dtype=float)

# rows: [pos_z, pos_r, pos_theta, vel_z, vel_r, vel_theta]
PARTICLES = [
    [1.5e-3, 1.5e-3, 0.0,    0.0,    0.0,   0.0],
    [1.5e-3, 1.5e-3, 0.0, 1000.0, -500.0, 300.0],
    [2.7e-3, 1.3e-3, 0.0,  500.0,  200.0,   0.0],
    [1.1e-3, 2.9e-3, 0.0, -200.0,  800.0,-100.0],
    [3.5e-3, 3.5e-3, 0.0,  300.0, -300.0, 150.0],
    [2.0e-3, 2.0e-3, 0.0,    0.0,    0.0,   0.0],
]


def serial_push(particles_data, efz, efr):
    results = []
    for p in particles_data:
        pos = list(p["pos"])
        vel = list(p["vel"])

        i = math.trunc(pos[0] / DZ)
        j = math.trunc(pos[1] / DR)
        di = pos[0] / DZ - i
        dj = pos[1] / DR - j

        ez = (efz[i][j]   * (1-di)*(1-dj) + efz[i+1][j]   * di*(1-dj) +
              efz[i][j+1] * (1-di)*dj     + efz[i+1][j+1] * di*dj)
        er = (efr[i][j]   * (1-di)*(1-dj) + efr[i+1][j]   * di*(1-dj) +
              efr[i][j+1] * (1-di)*dj     + efr[i+1][j+1] * di*dj)

        vel[0] += QM * ez * DT;  pos[0] += vel[0] * DT
        vel[1] += QM * er * DT;  pos[1] += vel[1] * DT
        vel[2] += 0.0;           pos[2] += vel[2] * DT

        r = math.sqrt(pos[1]**2 + pos[2]**2)
        if r > 0.0:
            sin_t = pos[2] / r
            cos_t = math.sqrt(max(0.0, 1.0 - sin_t**2))
            pos[1] = r;  pos[2] = 0.0
            u2 = cos_t * vel[1] - sin_t * vel[2]
            v2 = sin_t * vel[1] + cos_t * vel[2]
            vel[1] = u2;  vel[2] = v2

        results.append({"pos": pos, "vel": vel})
    return results


def parallel_push(particles_data, efz, efr):
    # convert to (N, 6) numpy array
    arr = numpy.array(
        [[*p["pos"], *p["vel"]] for p in particles_data], dtype=numpy.float64)

    # put fields in shared memory
    efz_c = numpy.ascontiguousarray(efz, dtype=numpy.float64)
    efr_c = numpy.ascontiguousarray(efr, dtype=numpy.float64)
    s_efz = shm_module.SharedMemory(create=True, size=efz_c.nbytes)
    s_efr = shm_module.SharedMemory(create=True, size=efr_c.nbytes)
    numpy.ndarray(efz_c.shape, dtype=numpy.float64, buffer=s_efz.buf)[:] = efz_c
    numpy.ndarray(efr_c.shape, dtype=numpy.float64, buffer=s_efr.buf)[:] = efr_c

    chunks = numpy.array_split(arr, N_CORES)
    args = [(c, s_efz.name, s_efr.name, efz_c.shape, QM, DT, DZ, DR) for c in chunks]
    with Pool(N_CORES) as pool:
        result_arr = numpy.concatenate(pool.map(push_chunk_shm, args))

    s_efz.close(); s_efz.unlink()
    s_efr.close(); s_efr.unlink()

    return [{"pos": list(row[:3]), "vel": list(row[3:])} for row in result_arr]


def assert_equal(s, p, tol=1e-12):
    for idx, (a, b) in enumerate(zip(s, p)):
        for dim in range(3):
            assert math.isclose(a["pos"][dim], b["pos"][dim], rel_tol=tol, abs_tol=1e-30), \
                f"particle {idx} pos[{dim}]: serial={a['pos'][dim]:.15e}  parallel={b['pos'][dim]:.15e}"
            assert math.isclose(a["vel"][dim], b["vel"][dim], rel_tol=tol, abs_tol=1e-30), \
                f"particle {idx} vel[{dim}]: serial={a['vel'][dim]:.15e}  parallel={b['vel'][dim]:.15e}"


# serial and parallel must give identical results for all 6 particles
def test_serial_vs_parallel():
    data = [{"pos": row[:3], "vel": row[3:]} for row in PARTICLES]
    assert_equal(serial_push(data, EFZ, EFR), parallel_push(data, EFZ, EFR))


# a stationary particle in a zero field must not move at all
def test_no_force_no_motion():
    zero = numpy.zeros((6, 6))
    data = [{"pos": [2.3e-3, 1.7e-3, 0.0], "vel": [0.0, 0.0, 0.0]}]
    result = parallel_push(data, zero, zero)
    for dim in range(3):
        assert result[0]["pos"][dim] == data[0]["pos"][dim]
        assert result[0]["vel"][dim] == 0.0


# uniform field means every particle gets the same delta-v regardless of position
def test_uniform_field_same_dv():
    efz_uniform = numpy.full((6, 6), 5000.0)
    data = [
        {"pos": [1.2e-3, 1.2e-3, 0.0], "vel": [0.0, 0.0, 0.0]},
        {"pos": [2.5e-3, 2.5e-3, 0.0], "vel": [0.0, 0.0, 0.0]},
        {"pos": [3.8e-3, 1.8e-3, 0.0], "vel": [0.0, 0.0, 0.0]},
    ]
    expected_dv = QM * 5000.0 * DT
    for r in parallel_push(data, efz_uniform, numpy.zeros((6, 6))):
        assert math.isclose(r["vel"][0], expected_dv, rel_tol=1e-12)


# after the push the theta component of position must always come back to 0
def test_theta_zeroed_after_rotation():
    data = [{"pos": [2.0e-3, 1.5e-3, 0.0], "vel": [0.0, 200.0, 1000.0]}]
    result = parallel_push(data, EFZ, EFR)
    assert math.isclose(result[0]["pos"][2], 0.0, abs_tol=1e-30)
    assert result[0]["pos"][1] > 0.0


# particle on node (2,3) must gather exactly efz[2][3]=1200 and efr[2][3]=-600
def test_on_node_gather():
    data = [{"pos": [2 * DZ, 3 * DR, 0.0], "vel": [0.0, 0.0, 0.0]}]
    result = parallel_push(data, EFZ, EFR)
    assert math.isclose(result[0]["vel"][0], QM * 1200.0 * DT, rel_tol=1e-12)
    assert math.isclose(result[0]["vel"][1], QM * -600.0 * DT, rel_tol=1e-12)


# reversing the particle list must not change the per-particle outcome
def test_order_doesnt_matter():
    data = [{"pos": row[:3], "vel": row[3:]} for row in PARTICLES]
    r1 = parallel_push(data, EFZ, EFR)
    r2 = list(reversed(parallel_push(list(reversed(data)), EFZ, EFR)))
    assert_equal(r1, r2)


if __name__ == "__main__":
    tests = [
        test_serial_vs_parallel,
        test_no_force_no_motion,
        test_uniform_field_same_dv,
        test_theta_zeroed_after_rotation,
        test_on_node_gather,
        test_order_doesnt_matter,
    ]
    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"PASSED  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAILED  {t.__name__}: {e}")
            failed += 1
    print(f"\n{passed}/{len(tests)} passed")
    if failed:
        sys.exit(1)

# Parameterizable version of rz-pic.py
# Reads RZ_NZ, RZ_NR, RZ_TS from environment variables for benchmarking.
import os
import numpy
import math
from random import seed, random

# ---- read parameters from env (with defaults matching the original) ----
nz = int(os.environ.get("RZ_NZ", 35))
nr = int(os.environ.get("RZ_NR", 12))
TIMESTEPS = int(os.environ.get("RZ_TS", 1000))

dz = 1e-3
dr = 1e-3
dt = 5e-9

QE = 1.602e-19
AMU = 1.661e-27
EPS0 = 8.854e-12

charge = QE
m = 40 * AMU
qm = charge / m
spwt = 50

n0 = 1e12
phi0 = 100
phi1 = 0
kTe = 5


def XtoL(pos):
    return [pos[0] / dz, pos[1] / dr]


def Pos(lc):
    return [lc[0] * dz, lc[1] * dr]


def R(j):
    return j * dr


def gather(data, lc):
    i = math.trunc(lc[0])
    j = math.trunc(lc[1])
    di = lc[0] - i
    dj = lc[1] - j
    return (data[i][j] * (1 - di) * (1 - dj) +
            data[i + 1][j] * di * (1 - dj) +
            data[i][j + 1] * (1 - di) * dj +
            data[i + 1][j + 1] * di * dj)


def scatter(data, lc, value):
    i = int(numpy.trunc(lc[0]))
    j = int(numpy.trunc(lc[1]))
    di = lc[0] - i
    dj = lc[1] - j
    data[i][j] += (1 - di) * (1 - dj) * value
    data[i + 1][j] += di * (1 - dj) * value
    data[i][j + 1] += (1 - di) * dj * value
    data[i + 1][j + 1] += di * dj * value


class Particle:
    def __init__(self, pos, vel):
        self.pos = [pos[0], pos[1], 0]
        self.vel = [vel[0], vel[1], vel[2]]


def sampleIsotropicVel(vth):
    theta = 2 * math.pi * random()
    Rv = -1.0 + 2 * random()
    a = math.sqrt(1 - Rv * Rv)
    n = (math.cos(theta) * a, math.sin(theta) * a, Rv)
    vm = numpy.zeros(3)
    vm[0:3] = math.sqrt(2) * vth * (2 * (random() + random() + random() - 1.5))
    return (n[0] * vm[0], n[1] * vm[1], n[2] * vm[2])


def solvePotential(phi, cell_type, rho_i, r, max_it=100):
    P = numpy.copy(phi)
    g = numpy.zeros_like(phi)
    dz2 = dz * dz
    dr2 = dr * dr

    for it in range(max_it):
        rho_e = QE * n0 * numpy.exp(numpy.subtract(P, phi0) / kTe)
        b = numpy.where(cell_type <= 0, (rho_i - rho_e) / EPS0, 0)

        g[1:-1, 1:-1] = (b[1:-1, 1:-1] +
                         (phi[1:-1, 2:] + phi[1:-1, :-2]) / dr2 +
                         (phi[1:-1, 0:-2] - phi[1:-1, 2:]) / (2 * dr * r[1:-1, 1:-1]) +
                         (phi[2:, 1:-1] + phi[:-2, 1:-1]) / dz2) / (2 / dr2 + 2 / dz2)

        g[0] = g[1]
        g[-1] = g[-2]
        g[:, -1] = g[:, -2]
        g[:, 0] = g[:, 1]

        phi = numpy.where(cell_type > 0, P, g)
    return phi


def computeEF(phi, efz, efr):
    efz[1:-1] = (phi[0:nz - 2] - phi[2:nz + 1]) / (2 * dz)
    efr[:, 1:-1] = (phi[:, 0:nr - 2] - phi[:, 2:nr + 1]) / (2 * dr)
    efz[0, :] = (phi[0, :] - phi[1, :]) / dz
    efz[-1, :] = (phi[-2, :] - phi[-1, :]) / dz
    efr[:, 0] = (phi[:, 0] - phi[:, 1]) / dr
    efr[:, -1] = (phi[:, -2] - phi[:, -1]) / dr


def main():
    global n0

    seed()

    phi = numpy.zeros([nz, nr])
    efz = numpy.zeros([nz, nr])
    efr = numpy.zeros([nz, nr])
    rho_i = numpy.zeros([nz, nr])
    den = numpy.zeros([nz, nr])
    cell_type = numpy.zeros([nz, nr])

    tube1_radius = 6 * dr
    tube1_length = 0.01
    tube1_aperture_rad = 4 * dr
    tube2_radius = tube1_radius + dr
    tube2_length = tube1_length + 2 * dz
    tube2_aperture_rad = 3 * dr
    tube_i_max, tube_j_max = map(int, XtoL([4 * dz, tube1_radius]))

    for i in range(nz):
        for j in range(nr):
            pos = Pos([i, j])
            if ((i == 0 and pos[1] < tube1_radius) or
                    (pos[0] <= tube1_length and pos[1] >= tube1_radius and pos[1] < tube1_radius + 0.5 * dr) or
                    (pos[0] >= tube1_length and pos[0] < tube1_length + 0.5 * dz and
                     pos[1] >= tube1_aperture_rad and pos[1] < tube1_radius)):
                cell_type[i][j] = 1
                phi[i][j] = phi0
            if ((pos[0] <= tube2_length and pos[1] >= tube2_radius and pos[1] < tube2_radius + 0.5 * dr) or
                    (pos[0] >= tube2_length and pos[0] <= tube2_length + 1.5 * dz and
                     pos[1] >= tube2_aperture_rad and pos[1] <= tube2_radius)):
                cell_type[i][j] = 2
                phi[i][j] = phi1

    node_volume = numpy.zeros([nz, nr])
    for i in range(nz):
        for j in range(nr):
            j_min = max(j - 0.5, 0)
            j_max = min(j + 0.5, nr - 1)
            a = 0.5 if (i == 0 or i == nz - 1) else 1.0
            node_volume[i][j] = a * dz * (R(j_max) ** 2 - R(j_min) ** 2)

    # radial distances for solver
    r = numpy.zeros([nz, nr])
    for i in range(nz):
        for j in range(nr):
            r[i][j] = R(j)

    particles = []
    mpf_rem = numpy.zeros([nz, nr])
    rho_i = numpy.zeros([nz, nr])

    phi = solvePotential(phi, cell_type, rho_i, r, 1000)
    computeEF(phi, efz, efr)

    for ts in range(TIMESTEPS + 1):
        den = numpy.zeros([nz, nr])

        na = 1e15
        ne = 1e12
        k = 2e-10
        dni = k * ne * na * dt

        for i in range(1, tube_i_max):
            for j in range(0, tube_j_max):
                if cell_type[i][j] > 0:
                    continue
                cell_volume = gather(node_volume, (i + 0.5, j + 0.5))
                mpf_new = dni * cell_volume / spwt + mpf_rem[i][j]
                mp_new = int(math.trunc(mpf_new + random()))
                mpf_rem[i][j] = mpf_new - mp_new
                for p in range(mp_new):
                    pos = Pos([i + random(), j + random()])
                    vel = sampleIsotropicVel(300)
                    particles.append(Particle(pos, vel))

        for part in particles:
            lc = XtoL(part.pos)
            part_ef = [gather(efz, lc), gather(efr, lc), 0]
            for dim in range(3):
                part.vel[dim] += qm * part_ef[dim] * dt
                part.pos[dim] += part.vel[dim] * dt

            r_new = math.sqrt(part.pos[1] ** 2 + part.pos[2] ** 2)
            sin_theta_r = part.pos[2] / r_new
            part.pos[1] = r_new
            part.pos[2] = 0
            cos_theta_r = math.sqrt(1 - sin_theta_r ** 2)
            u2 = cos_theta_r * part.vel[1] - sin_theta_r * part.vel[2]
            v2 = sin_theta_r * part.vel[1] + cos_theta_r * part.vel[2]
            part.vel[1] = u2
            part.vel[2] = v2

        p = 0
        np_count = len(particles)
        while p < np_count:
            part = particles[p]
            lc = XtoL(part.pos)
            i = int(numpy.trunc(lc[0]))
            j = int(numpy.trunc(lc[1]))
            if i < 0 or i >= nz - 1 or j >= nr - 1 or cell_type[i][j] > 0:
                particles[p] = particles[np_count - 1]
                np_count -= 1
                continue
            scatter(den, lc, spwt)
            p += 1

        particles = particles[0:np_count]
        den /= node_volume
        rho_i = charge * den

        phi = solvePotential(phi, cell_type, rho_i, r)
        computeEF(phi, efz, efr)
        n0 = max(den.max(), 1e-10)


if __name__ == "__main__":
    main()

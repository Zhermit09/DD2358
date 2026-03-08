# axisymmetric (RZ) particle in cell code example
#
# see https://www.particleincell.com/2015/rz-pic/ for more info
# simulates a simplistic ion source in which ions are
# produced from a volumetric source with constant electron and 
# neutral density (i.e. not taking into account avalanche ionization
# or source depletion)
#
# code illustrates velocity and position rotation in RZ
#
# requires numpy, scipy, pylab, and mathplotlib

import math
import numpy
import torch as th
# import pylab as pl
from random import (seed, random)


def XtoL(pos):
    lc = [pos[0] / dz, pos[1] / dr]
    return lc


def Pos(lc):
    pos = [lc[0] * dz, lc[1] * dr]
    return pos


def R(j):
    return j * dr


def gather(data, lc0, lc1):
    i = int(lc0)
    j = int(lc1)
    di = lc0 - i
    dj = lc1 - j

    ip = i + 1
    jp = j + 1

    a = data[i, j]
    b = data[ip, j]
    c = data[i, jp]
    d = data[ip, jp]

    x0 = a + di * (b - a)
    x1 = c + di * (d - c)

    return x0 + dj * (x1 - x0)


def gather_init(data, lc0, lc1):
    a = data[0: -1, 0:-1]
    b = data[1:, 0:-1]
    c = data[0: -1, 1:]
    d = data[1:, 1:]

    di = lc0
    dj = lc1

    x0 = a + di * (b - a)
    x1 = c + dj * (d - c)

    return x0 + dj * (x1 - x0)


def scatter(data, lc0, lc1, value):
    i = int(lc0)
    j = int(lc1)
    di = lc0 - i
    dj = lc1 - j

    ip = i + 1
    jp = j + 1

    di1 = 1 - di
    dj1 = 1 - dj

    v1 = dj1 * value
    v2 = dj * value

    data[i, j] += di1 * v1
    data[ip, j] += di * v1
    data[i, jp] += di1 * v2
    data[ip, jp] += di * v2


# particle definition
class Particle:
    def __init__(self, pos, vel):
        self.pos = [pos[0], pos[1], 0]  # xyz position
        self.vel = [vel[0], vel[1], vel[2]]


# --- helper functions ----
def sampleIsotropicVel(vth):
    # pick a random angle
    theta = 2 * math.pi * random()

    # pick a random direction for n[2]
    R = -1.0 + 2 * random()
    a = math.sqrt(1 - R * R)
    n = (math.cos(theta) * a, math.sin(theta) * a, R)

    # pick maxwellian velocities
    vm = numpy.zeros(3)
    vm[0:3] = math.sqrt(2) * vth * (2 * (random() + random() + random() - 1.5))

    vel = (n[0] * vm[0], n[1] * vm[1], n[2] * vm[2])
    return vel


# simple Jacobian solver, does not do any convergence checking
def solvePotential(phi, max_it=100):
    # device = th.device("cpu")
    phi_t = th.from_numpy(phi).cuda()
    cell_type_t = th.from_numpy(cell_type).cuda()
    rho_i_t = th.from_numpy(rho_i).cuda()

    dz2 = dz * dz
    dr2 = dr * dr
    dr_x2 = 2 * dr
    denom = 2.0 / dr2 + 2.0 / dz2

    g = th.empty_like(phi_t, dtype=th.float64)
    b = th.empty_like(phi_t, dtype=th.float64)
    mask = cell_type_t <= 0

    # compute electron term
    rho_e = QE * n0 * th.exp((phi_t[mask] - phi0) / kTe)
    b[mask] = (rho_i_t[mask] - rho_e) / EPS0

    # set radia
    row = th.tensor([dr_x2 * R(j) for j in range(nr)], dtype=th.float64, device=th.device("cuda"))
    r = row.unsqueeze(0).expand(nz, nr)

    g_ij = g[1:-1, 1:-1]
    b_ij = b[1:-1, 1:-1]
    r_ij = r[1:-1, 1:-1]
    phi_ijp = phi_t[1:-1, 2:]
    phi_ijm = phi_t[1:-1, :-2]
    phi_ipj = phi_t[2:, 1:-1]
    phi_imj = phi_t[:-2, 1:-1]

    g_i0 = g[:, 0]
    g_i1 = g[:, 1]
    g_im1 = g[:, -1]
    g_im2 = g[:, -2]
    g_0j = g[0]
    g_1j = g[1]
    g_m1j = g[-1]
    g_m2j = g[-2]

    for it in range(max_it):
        # regular form inside
        g_ij[...] = (
                            b_ij +
                            (phi_ijm + phi_ijp) / dr2 +
                            (phi_ijm - phi_ijp) / r_ij +
                            (phi_ipj + phi_imj) / dz2
                    ) / denom

        # neumann boundaries
        g_i0[...] = g_i1  # left
        g_im1[...] = g_im2  # right
        g_0j[...] = g_1j  # top
        g_m1j[...] = g_m2j  # bottom

        # dirichlet nodes
        phi_t[mask] = g[mask]

    return phi_t.cpu().numpy()


# computes electric field
def computeEF(phi, efz, efr):
    # central difference, not right on walls
    efz[1:-1] = (phi[0:nz - 2] - phi[2:nz + 1]) / (2 * dz)
    efr[:, 1:-1] = (phi[:, 0:nr - 2] - phi[:, 2:nr + 1]) / (2 * dr)

    # one-sided difference on boundaries
    efz[0, :] = (phi[0, :] - phi[1, :]) / dz
    efz[-1, :] = (phi[-2, :] - phi[-1, :]) / dz
    efr[:, 0] = (phi[:, 0] - phi[:, 1]) / dr
    efr[:, -1] = (phi[:, -2] - phi[:, -1]) / dr


draw_plot = False
# allocate memory space
nr = 6
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

phi = numpy.zeros([nz, nr])
efz = numpy.zeros([nz, nr])
efr = numpy.zeros([nz, nr])
rho_i = numpy.zeros([nz, nr])
den = numpy.zeros([nz, nr])

# ---- sugarcube domain --------------------
cell_type = numpy.zeros([nz, nr])
tube1_radius = (nr / 2) * dr
tube1_length = 0.28 * nz * dz
tube1_aperture_rad = (nr / 3) * dr
tube2_radius = tube1_radius + dr
tube2_length = tube1_length + 2 * dz
tube2_aperture_rad = (nr / 4) * dr
[tube_i_max, tube_j_max] = map(int, XtoL([4 * dz, tube1_radius]))


def reassign_globals(NR=12):
    global nr, nz, n0
    global phi, efz, efr, rho_i, den
    global cell_type, tube_i_max, tube_j_max
    global tube1_radius, tube1_length, tube1_aperture_rad
    global tube2_radius, tube2_length, tube2_aperture_rad

    nr = NR
    nz = nr * 3
    n0 = 1e12

    phi = numpy.zeros([nz, nr])
    efz = numpy.zeros([nz, nr])
    efr = numpy.zeros([nz, nr])
    rho_i = numpy.zeros([nz, nr])
    den = numpy.zeros([nz, nr])

    # ---- sugarcube domain --------------------
    cell_type = numpy.zeros([nz, nr])
    tube1_radius = (nr / 2) * dr
    tube1_length = 0.28 * nz * dz
    tube1_aperture_rad = (nr / 3) * dr
    tube2_radius = tube1_radius + dr
    tube2_length = tube1_length + 2 * dz
    tube2_aperture_rad = (nr / 4) * dr
    [tube_i_max, tube_j_max] = map(int, XtoL([4 * dz, tube1_radius]))


# @profile
def main():
    global nz, nr, dz, dr, dt
    global QE, AMU, EPS0
    global charge, m, qm, spwt
    global n0, phi0, phi1, kTe
    global phi, efz, efr, rho_i, den

    # ---------- INITIALIZATION ----------------------------------------

    # pl.close('all')
    # seed()

    for i in range(0, nz):
        for j in range(0, nr):
            pos = Pos([i, j])  # node position

            # inner tube
            if ((i == 0 and pos[1] < tube1_radius) or
                    (pos[0] <= tube1_length and tube1_radius <= pos[1] < tube1_radius + 0.5 * dr) or
                    (tube1_length <= pos[0] < tube1_length + 0.5 * dz and
                     tube1_aperture_rad <= pos[1] < tube1_radius)):
                cell_type[i][j] = 1
                phi[i][j] = phi0

            if ((pos[0] <= tube2_length and tube2_radius <= pos[1] < tube2_radius + 0.5 * dr) or
                    (tube2_length <= pos[0] <= tube2_length + 1.5 * dz and
                     tube2_aperture_rad <= pos[1] <= tube2_radius)):
                cell_type[i][j] = 2
                phi[i][j] = phi1

    # ----------- COMPUTE NODE VOLUMES ------------------------
    node_volume = numpy.zeros([nz, nr])
    for i in range(0, nz):
        for j in range(0, nr):
            j_min = j - 0.5
            j_max = j + 0.5
            if j_min < 0: j_min = 0
            if j_max > nr - 1: j_max = nr - 1
            a = 0.5 if (i == 0 or i == nz - 1) else 1.0
            # note, this is r*dr for non-boundary nodes
            node_volume[i][j] = a * dz * (R(j_max) ** 2 - R(j_min) ** 2)

        # create an array of particles
    particles = []

    # counter for fractional particles
    mpf_rem = numpy.zeros([nz, nr])
    rho_i = numpy.zeros([nz, nr])

    lambda_d = math.sqrt(EPS0 * kTe / (n0 * QE))
    # print("Debye length is %.4g, which is %.2g*dz" % (lambda_d, lambda_d / dz))
    # print("Expected ion speed is %.2f m/s" % math.sqrt(2 * phi0 * qm))

    # positions for plotting
    pos_r = numpy.linspace(0, (nr - 1) * dr, nr)
    pos_z = numpy.linspace(0, (nz - 1) * dz, nz)

    # solve potential
    # print(type(phi))
    phi = solvePotential(phi, 1000)
    # print(type(phi))
    computeEF(phi, efz, efr)

    lc = numpy.full((nz - 1, nr - 1), 0.5)
    cell_volume = gather_init(node_volume, lc, lc)[1: tube_i_max, 0:tube_j_max]
    # ----------- MAIN LOOP --------------------------------------------
    for ts in range(0, 1000 + 1):

        den = numpy.zeros([nz, nr])

        # compute production rate
        na = 1e15
        ne = 1e12
        k = 2e-10  # not a physical value
        dni = k * ne * na * dt

        # inject particles
        for i in range(1, tube_i_max):
            for j in range(0, tube_j_max):

                # skip over solid cells
                if cell_type[i][j] > 0: continue

                # interpolate node volume to cell center to get cell volume
                # cell_volume = gather(node_volume, i + 0.5, j + 0.5)

                # floating point production rate
                mpf_new = dni * cell_volume[i - 1, j] / spwt + mpf_rem[i][j]

                # truncate down, adding randomness
                mp_new = int(mpf_new + random())

                # save fraction part
                mpf_rem[i][j] = mpf_new - mp_new  # new fractional reminder

                # generate this many particles
                for p in range(mp_new):
                    pos = Pos([i + random(), j + random()])
                    vel = sampleIsotropicVel(300)
                    particles.append(Particle(pos, vel))

        # some arbitrary min value
        max_zvel = 0

        # push particles
        for part in particles:
            # gather electric field
            lc0, lc1 = XtoL(part.pos)
            part_ef = [gather(efz, lc0, lc1), gather(efr, lc0, lc1), 0]
            for dim in range(3):
                part.vel[dim] += qm * part_ef[dim] * dt
                part.pos[dim] += part.vel[dim] * dt

            # get new maximum velocity, for screen output
            if part.vel[0] > max_zvel: max_zvel = part.vel[0]

            # rotate particle back to ZR plane
            r = math.sqrt(part.pos[1] * part.pos[1] + part.pos[2] * part.pos[2])
            sin_theta_r = part.pos[2] / r
            part.pos[1] = r
            part.pos[2] = 0

            # rotate velocity
            cos_theta_r = math.sqrt(1 - sin_theta_r * sin_theta_r)
            u2 = cos_theta_r * part.vel[1] - sin_theta_r * part.vel[2]
            v2 = sin_theta_r * part.vel[1] + cos_theta_r * part.vel[2]
            part.vel[1] = u2
            part.vel[2] = v2

        # compute density
        p = 0
        np = len(particles)
        while p < np:

            part = particles[p]
            lc0, lc1 = XtoL(part.pos)
            i = int(lc0)
            j = int(lc1)

            #
            if i < 0 or i >= nz - 1 or j >= nr - 1 or cell_type[i][j] > 0:
                # replace current data with the last entry
                particles[p] = particles[np - 1]
                np -= 1
                continue

            scatter(den, lc0, lc1, spwt)
            p += 1

        # resize particle array
        particles = particles[0:np]

        # divide by node volume
        den /= node_volume
        rho_i = charge * den

        # update potential
        phi = solvePotential(phi)

        # compute electric field
        computeEF(phi, efz, efr)

        # recompute reference density
        n0 = den.max()


if __name__ == "__main__":
    main()

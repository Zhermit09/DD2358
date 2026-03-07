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

import numpy
import pylab as pl
import math
from random import (seed, random)

cimport cython
# noinspection PyUnresolvedReferences
cimport numpy
from libc.math cimport exp

numpy.import_array()

cdef XtoL(pos):
    lc = [pos[0] / dz, pos[1] / dr]
    return lc

cdef Pos(lc):
    pos = [lc[0] * dz, lc[1] * dr]
    return pos

@cython.wraparound(False)
@cython.cdivision(True)
cdef  inline double R(double j) noexcept:
    return j * dr

@cython.wraparound(False)
@cython.cdivision(True)
cdef inline double gather(double[:, :] data, double lc0, double lc1) noexcept:
    cdef int i = <int> lc0
    cdef int j = <int> lc1
    cdef double di = lc0 - i
    cdef double dj = lc1 - j

    cdef int ip = i + 1
    cdef int jp = j + 1

    cdef double a = data[i, j]
    cdef double b = data[ip, j]
    cdef double c = data[i, jp]
    cdef double d = data[ip, jp]

    cdef double x0 = a + di * (b - a)
    cdef double x1 = c + di * (d - c)

    return x0 + dj * (x1 - x0)

@cython.wraparound(False)
@cython.cdivision(True)
cdef inline void scatter(double[:, :] data, double lc0, double lc1, double value) noexcept:
    cdef int i = <int> lc0
    cdef int j = <int> lc1
    cdef double di = lc0 - i
    cdef double dj = lc1 - j

    cdef int  ip = i + 1
    cdef int  jp = j + 1

    cdef double  di1 = 1 - di
    cdef double  dj1 = 1 - dj

    cdef double v1 = dj1 * value
    cdef double v2 = dj * value

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
cdef sampleIsotropicVel(vth):
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
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline void solvePotential(double[:, :] phi, int max_it=100) noexcept:
    cdef int i, j, it
    cdef double dz2 = dz * dz
    cdef double dr2 = dr * dr
    cdef double dr_x2 = 2 * dr
    cdef double denom = 2.0 / dr2 + 2.0 / dz2

    cdef double[:, :] cell_type_v = cell_type
    cdef double[:, :] rho_i_v = rho_i
    cdef double[:, :] P = numpy.copy(phi)
    cdef double[:, :] g = numpy.empty_like(phi)

    # set radia
    cdef double[:] r = numpy.empty((nr,))
    for j in range(nr):
        r[j] = dr_x2 * R(j)

    # compute electron term
    cdef double[:, :] b = numpy.zeros_like(phi)
    for i in range(nz):
        for j in range(nr):
            if cell_type_v[i, j] > 0: continue
            b[i, j] = (rho_i_v[i, j] - QE * n0 * exp((P[i, j] - phi0) / kTe)) / EPS0

    cdef double phi_ijp, phi_ijm, phi_ipj, phi_imj
    for it in range(max_it):
        # regular form inside
        for i in range(1, nz - 1):
            for j in range(1, nr - 1):
                phi_ijp = phi[i, j + 1]
                phi_ijm = phi[i, j - 1]
                phi_ipj = phi[i + 1, j]
                phi_imj = phi[i - 1, j]
                g[i, j] = (
                                  b[i, j] +
                                  (phi_ijm + phi_ijp) / dr2 +
                                  (phi_ijm - phi_ijp) / r[j] +
                                  (phi_ipj + phi_imj) / dz2
                          ) / denom
            # neumann boundaries (L+R)
            g[i, 0] = g[i, 1]
            g[i, nr - 1] = g[i, nr - 2]

        # neumann boundaries (T+B)
        for j in range(nr):
            g[0, j] = g[1, j]
            g[nz - 1, j] = g[nz - 2, j]

        for i in range(nz):
            for j in range(nr):
                phi[i, j] = P[i, j] if cell_type_v[i, j] > 0 else g[i, j]

# computes electric field
cdef computeEF(phi, efz, efr):
    # central difference, not right on walls
    efz[1:-1] = (phi[0:nz - 2] - phi[2:nz + 1]) / (2 * dz)
    efr[:, 1:-1] = (phi[:, 0:nr - 2] - phi[:, 2:nr + 1]) / (2 * dr)

    # one-sided difference on boundaries
    efz[0, :] = (phi[0, :] - phi[1, :]) / dz
    efz[-1, :] = (phi[-2, :] - phi[-1, :]) / dz
    efr[:, 0] = (phi[:, 0] - phi[:, 1]) / dr
    efr[:, -1] = (phi[:, -2] - phi[:, -1]) / dr

cdef plot(ax, data, pos_z, pos_r, scatter=False):
    pl.sca(ax)
    pl.cla()
    cf = pl.contourf(pos_z, pos_r, numpy.transpose(data), 8, alpha=.75, cmap='jet')
    # cf = pl.pcolormesh(pos_z, pos_r, numpy.transpose(data))
    if scatter:
        # ax.hold(True);
        (ZZ, RR) = pl.meshgrid(pos_z, pos_r)
        ax.scatter(ZZ, RR, c=numpy.transpose(cell_type), cmap='jet')
    ax.set_yticks(pos_r)
    ax.set_xticks(pos_z)
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    pl.xlim(min(pos_z), max(pos_z))
    pl.ylim(min(pos_r), max(pos_r))
    ax.grid(True, which='both', color='k', linestyle='-')
    ax.set_aspect('equal', adjustable='box')

#  pl.colorbar(cf,ax=pl.gca(),orientation='horizontal',shrink=0.75, pad=0.01)

cdef bool draw_plot = False

# allocate memory space
cdef int nr = 12
cdef int nz = nr * 3
cdef const double dz = 1e-3
cdef const double dr = 1e-3
cdef const double dt = 5e-9

cdef const double QE = 1.602e-19
cdef const double AMU = 1.661e-27
cdef const double EPS0 = 8.854e-12

cdef double charge = QE
cdef double m = 40 * AMU  # argon ions
cdef double qm = charge / m
cdef const int spwt = 50

# solver parameters
cdef double n0 = 1e12
cdef const int phi0 = 100
cdef const int phi1 = 0
cdef const int kTe = 5

phi = numpy.zeros([nz, nr])
efz = numpy.zeros([nz, nr])
efr = numpy.zeros([nz, nr])
rho_i = numpy.zeros([nz, nr])
den = numpy.zeros([nz, nr])

# ---- sugarcube domain --------------------
cell_type = numpy.zeros([nz, nr])
cdef double tube1_radius = (nr / 2) * dr
cdef double tube1_length = 0.28 * nz * dz
cdef double tube1_aperture_rad = (nr / 3) * dr
cdef double tube2_radius = tube1_radius + dr
cdef double tube2_length = tube1_length + 2 * dz
cdef double tube2_aperture_rad = (nr / 4) * dr
cdef int tube_i_max, tube_j_max
[tube_i_max, tube_j_max] = map(int, XtoL(numpy.array([4 * dz, tube1_radius])))

cpdef reassign_globals(NR=12):
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

cpdef main():
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
    # fig1 = pl.figure(num=None, figsize=(10, 10), dpi=80, facecolor='w', edgecolor='k')
    if draw_plot: sub = (pl.subplot(211), pl.subplot(212))

    # solve potential
    solvePotential(phi, 1000)
    computeEF(phi, efz, efr)

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
                cell_volume = gather(node_volume, i + 0.5, j + 0.5)

                # floating point production rate
                mpf_new = dni * cell_volume / spwt + mpf_rem[i][j]

                # truncate down, adding randomness
                mp_new = int(math.trunc(mpf_new + random()))

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
            lc = XtoL(part.pos)
            part_ef = [gather(efz, lc[0], lc[1]), gather(efr, lc[0], lc[1]), 0]
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
            lc = XtoL(part.pos)
            i = int(numpy.trunc(lc[0]))
            j = int(numpy.trunc(lc[1]))

            #
            if i < 0 or i >= nz - 1 or j >= nr - 1 or cell_type[i][j] > 0:
                # replace current data with the last entry
                particles[p] = particles[np - 1]
                np -= 1
                continue

            scatter(den, lc[0], lc[1], spwt)
            p += 1

        # resize particle array
        particles = particles[0:np]

        # divide by node volume
        den /= node_volume
        rho_i = charge * den

        # update potential
        solvePotential(phi)

        # compute electric field
        computeEF(phi, efz, efr)

        # recompute reference density
        n0 = den.max()

        if draw_plot and ts % 10 == 0:
            # print("ts: %d, np: %d, phi range: %.2g:%.2g, max_den: %.3g, max_zvel: %.f" % (ts, len(particles), phi.min(),
            #                                                                              phi.max(), n0, max_zvel))

            # sub = pl.subplot(111,aspect='equal')

            # sub[0].hold(False)
            plot(sub[0], numpy.log10(numpy.where(den <= 1e4, 1e4, den)), pos_z, pos_r, scatter=False)
            plot(sub[1], phi, pos_z, pos_r)
            pl.draw()
            pl.pause(1e-4)  # allow for repaint

    # ----------- END OF MAIN LOOP ------------------------
    if draw_plot:
        plot(sub[0], numpy.log10(numpy.where(den <= 1e4, 1e4, den)), pos_z, pos_r, scatter=False)
        plot(sub[1], phi, pos_z, pos_r)
        # Q = pl.quiver(pos_z, pos_r, numpy.transpose(efz), numpy.transpose(efr),units='xy')
        pl.draw()

        # this will block execution until figure is closed
        pl.show()

if __name__ == "__main__":
    main()

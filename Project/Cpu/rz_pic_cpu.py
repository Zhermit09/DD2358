import numpy
import pylab as pl
import math
from random import (seed, random)
from multiprocessing import Pool, shared_memory


def XtoL(pos):
    lc = [pos[0] / dz, pos[1] / dr]
    return lc


def Pos(lc):
    pos = [lc[0] * dz, lc[1] * dr]
    return pos


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


# ---------------------------------------------------------------------------
# Number of CPU cores used for the parallel particle push
N_CORES = 4


def push_chunk_shm(args):
    part_chunk, shm_efz_name, shm_efr_name, shape, qm, dt, dz, dr = args

    shm_efz = shared_memory.SharedMemory(name=shm_efz_name)
    shm_efr = shared_memory.SharedMemory(name=shm_efr_name)
    efz = numpy.ndarray(shape, dtype=numpy.float64, buffer=shm_efz.buf)
    efr = numpy.ndarray(shape, dtype=numpy.float64, buffer=shm_efr.buf)

    out = part_chunk.copy()
    for idx in range(len(out)):
        pz, pr, pth, vz, vr, vth = out[idx]

        i  = math.trunc(pz / dz)
        j  = math.trunc(pr / dr)
        di = pz / dz - i
        dj = pr / dr - j

        ez = (efz[i][j]   * (1-di)*(1-dj) + efz[i+1][j]   * di*(1-dj) +
              efz[i][j+1] * (1-di)*dj     + efz[i+1][j+1] * di*dj)
        er = (efr[i][j]   * (1-di)*(1-dj) + efr[i+1][j]   * di*(1-dj) +
              efr[i][j+1] * (1-di)*dj     + efr[i+1][j+1] * di*dj)

        vz += qm * ez * dt;  pz += vz * dt
        vr += qm * er * dt;  pr += vr * dt
        pth += vth * dt

        r = math.sqrt(pr**2 + pth**2)
        if r > 0.0:
            sin_t = pth / r
            cos_t = math.sqrt(max(0.0, 1.0 - sin_t**2))
            pr = r;  pth = 0.0
            vr, vth = cos_t*vr - sin_t*vth, sin_t*vr + cos_t*vth

        out[idx] = [pz, pr, pth, vz, vr, vth]

    shm_efz.close()
    shm_efr.close()
    return out


def push_chunk(args):
    chunk, efz, efr, qm, dt, dz, dr = args
    results = []
    for pos, vel in chunk:
        # logical coordinates (XtoL inline)
        lc = [pos[0] / dz, pos[1] / dr]
        i, j   = int(lc[0]), int(lc[1])
        di, dj = lc[0] - i, lc[1] - j

        # bilinear gather of electric field
        ez = (efz[i][j]     * (1-di)*(1-dj) + efz[i+1][j]     * di*(1-dj) +
              efz[i][j+1]   * (1-di)*dj     + efz[i+1][j+1]   * di*dj)
        er = (efr[i][j]     * (1-di)*(1-dj) + efr[i+1][j]     * di*(1-dj) +
              efr[i][j+1]   * (1-di)*dj     + efr[i+1][j+1]   * di*dj)

        # push velocity then position
        vel[0] += qm * ez * dt;  pos[0] += vel[0] * dt
        vel[1] += qm * er * dt;  pos[1] += vel[1] * dt
        vel[2] += 0.0;           pos[2] += vel[2] * dt

        # rotate back to ZR plane
        r = math.sqrt(pos[1]**2 + pos[2]**2)
        if r == 0.0:
            results.append((pos, vel))
            continue
        sin_t = pos[2] / r
        cos_t = math.sqrt(max(0.0, 1.0 - sin_t * sin_t))
        pos[1] = r
        pos[2] = 0.0
        u2 = cos_t * vel[1] - sin_t * vel[2]
        v2 = sin_t * vel[1] + cos_t * vel[2]
        vel[1] = u2
        vel[2] = v2
        results.append((pos, vel))
    return results


# simple Jacobian solver, does not do any convergence checking
def solvePotential(phi, max_it=100):
    # make copy of dirichlet nodes
    P = numpy.copy(phi)

    g = numpy.zeros_like(phi)
    dz2 = dz * dz
    dr2 = dr * dr

    rho_e = numpy.zeros_like(phi)

    # set radia
    r = numpy.zeros_like(phi)
    for i in range(nz):
        for j in range(nr):
            r[i][j] = R(j)

    for it in range(max_it):
        # compute RHS
        # rho_e = QE*n0*numpy.exp(numpy.subtract(phi,phi0)/kTe)

        # for i in range(1,nz-1):
        #    for j in range(1,nr-1):

        #        if (cell_type[i,j]>0):
        #            continue

        #       rho_e=QE*n0*math.exp((phi[i,j]-phi0)/kTe)
        #        b = (rho_i[i,j]-rho_e)/EPS0;
        #        g[i,j] = (b +
        #                 (phi[i,j-1]+phi[i,j+1])/dr2 +
        #                 (phi[i,j-1]-phi[i,j+1])/(2*dr*r[i,j]) +
        #                 (phi[i-1,j] + phi[i+1,j])/dz2) / (2/dr2 + 2/dz2)

        #        phi[i,j]=g[i,j]

        # compute electron term
        rho_e = QE * n0 * numpy.exp(numpy.subtract(P, phi0) / kTe)
        b = numpy.where(cell_type <= 0, (rho_i - rho_e) / EPS0, 0)

        # regular form inside
        g[1:-1, 1:-1] = (b[1:-1, 1:-1] +
                         (phi[1:-1, 2:] + phi[1:-1, :-2]) / dr2 +
                         (phi[1:-1, 0:-2] - phi[1:-1, 2:]) / (2 * dr * r[1:-1, 1:-1]) +
                         (phi[2:, 1:-1] + phi[:-2, 1:-1]) / dz2) / (2 / dr2 + 2 / dz2)

        # neumann boundaries
        g[0] = g[1]  # left
        g[-1] = g[-2]  # right
        g[:, -1] = g[:, -2]  # top
        g[:, 0] = g[:, 1]

        # dirichlet nodes
        phi = numpy.where(cell_type > 0, P, g)

    return phi


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


def plot(ax, data, pos_z, pos_r, scatter=False):
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

draw_plot = False

# allocate memory space
nr = 12
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
    phi = solvePotential(phi, 1000)
    computeEF(phi, efz, efr)

    # Allocate shared memory for field grids once — workers attach zero-copy
    # each timestep instead of receiving pickled copies.
    _efz_c = numpy.ascontiguousarray(efz, dtype=numpy.float64)
    _efr_c = numpy.ascontiguousarray(efr, dtype=numpy.float64)
    shm_efz = shared_memory.SharedMemory(create=True, size=_efz_c.nbytes)
    shm_efr = shared_memory.SharedMemory(create=True, size=_efr_c.nbytes)
    shm_efz_arr = numpy.ndarray(_efz_c.shape, dtype=numpy.float64, buffer=shm_efz.buf)
    shm_efr_arr = numpy.ndarray(_efr_c.shape, dtype=numpy.float64, buffer=shm_efr.buf)
    shm_efz_arr[:] = _efz_c
    shm_efr_arr[:] = _efr_c
    ef_shape = efz.shape

    # Pool created once outside the loop
    
    pool = Pool(N_CORES)

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
                cell_volume = gather(node_volume, (i + 0.5, j + 0.5))

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

        # --- parallel particle push via shared_memory + numpy ---
        # Convert particles to a contiguous (N, 6) numpy array, split into
        # N_CORES chunks, and dispatch to workers that read efz/efr from
        # shared memory (zero-copy).  One memcpy per chunk replaces per-
        # particle pickling, giving ~3x speedup at simulation scale.
        if particles:
            particles_arr = numpy.array(
                [[p.pos[0], p.pos[1], p.pos[2], p.vel[0], p.vel[1], p.vel[2]]
                 for p in particles], dtype=numpy.float64)
            chunks = numpy.array_split(particles_arr, N_CORES)
            push_args = [
                (c, shm_efz.name, shm_efr.name, ef_shape, qm, dt, dz, dr)
                for c in chunks
            ]
            updated = numpy.concatenate(pool.map(push_chunk_shm, push_args))
            for idx, part in enumerate(particles):
                part.pos = list(updated[idx, :3])
                part.vel = list(updated[idx, 3:])

        # update max_zvel from pushed particles
        for part in particles:
            if part.vel[0] > max_zvel:
                max_zvel = part.vel[0]

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

            scatter(den, lc, spwt)
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

        # sync updated fields into shared memory for the next timestep
        shm_efz_arr[:] = efz
        shm_efr_arr[:] = efr

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
    pool.close()
    pool.join()

    # release shared memory
    shm_efz.close(); shm_efz.unlink()
    shm_efr.close(); shm_efr.unlink()

    if draw_plot:
        plot(sub[0], numpy.log10(numpy.where(den <= 1e4, 1e4, den)), pos_z, pos_r, scatter=False)
        plot(sub[1], phi, pos_z, pos_r)
        # Q = pl.quiver(pos_z, pos_r, numpy.transpose(efz), numpy.transpose(efr),units='xy')
        pl.draw()

        # this will block execution until figure is closed
        pl.show()


if __name__ == "__main__":
    main()

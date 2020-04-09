from .euclid cimport ceuclid
import cython
import numpy as np


def strain(r1, r2, r10, r20):
    return cstrain(r1, r2, r10, r20)


cdef inline double cstrain(double[:] r1, double[:] r2,
                           double[:] r10, double[:] r20):
    cdef double l, dl

    l = ceuclid(r1, r2)
    l0 = ceuclid(r10, r20)
    dl = l - l0

    return dl/l0


def strain2(l, r10, r20):
    return cstrain2(l, r10, r20)


cdef inline double cstrain2(double l, double[:] r10, double[:] r20):
    cdef double dl

    l0 = ceuclid(r10, r20)
    dl = l - l0

    return dl/l0


def family(double[:, :] r, double horizon):
    cdef int nnodes = r.shape[0]

    result = np.zeros(nnodes, dtype=np.intc)
    cdef int[:] result_view = result

    cdef int i, j

    for i in range(nnodes-1):
        for j in range(i+1, nnodes):
            if ceuclid(r[i], r[j]) < horizon:
                result_view[i] = result_view[i] + 1
                result_view[j] = result_view[j] + 1

    return result


def create_neighbour_list(double[:, :] r, double horizon, int size):
    cdef int nnodes = r.shape[0]

    result = np.zeros((nnodes, size), dtype=np.intc)
    cdef int[:, :] result_view = result
    n_neigh = np.zeros(nnodes, dtype=np.intc)
    cdef int[:] n_neigh_view = n_neigh

    cdef int i, j

    for i in range(nnodes-1):
        for j in range(i+1, nnodes):
            if ceuclid(r[i], r[j]) < horizon:
                # Add j as a neighbour of i
                result_view[i, n_neigh_view[i]] = j
                n_neigh_view[i] = n_neigh_view[i] + 1
                # Add i as a neighbour of j
                result_view[j, n_neigh_view[j]] = i
                n_neigh_view[j] = n_neigh_view[j] + 1

    return result, n_neigh


def break_bonds(double[:, :] r, double[:, :]r0, int[:, :] nlist,
                int[:] n_neigh, double critical_strain):
    cdef int nnodes = nlist.shape[0]

    cdef int i, j, i_n_neigh, neigh

    # Check neighbours for each node
    for i in range(nnodes):
        # Get current number of neighbours
        i_n_neigh = n_neigh[i]

        neigh = 0
        while neigh < i_n_neigh:
            j = nlist[i, neigh]

            if abs(cstrain(r[i], r[j], r0[i], r0[j])) < critical_strain:
                # Move onto the next neighbour
                neigh += 1
            else:
                # Remove this neighbour by replacing it with the last neighbour
                # on the list, then reducing the number of neighbours by 1
                nlist[i, neigh] = nlist[i, i_n_neigh-1]
                i_n_neigh -= 1

        n_neigh[i] = i_n_neigh


def damage(int[:] n_neigh, int[:] family):
    cdef nnodes = family.shape[0]

    result = np.empty(nnodes, dtype=np.float64)
    cdef double[:] result_view = result

    cdef double ifamily
    cdef int i

    for i in range(nnodes):
        ifamily = family[i]
        result_view[i] = (ifamily - n_neigh[i])/ifamily

    return result


def bond_force(double[:, :] r, double[:, :] r0, int[:, :] nlist,
               int[:] n_neigh, double[:] volume, double bond_stiffness):
    cdef int nnodes = nlist.shape[0]

    force = np.zeros((nnodes, 3), dtype=np.float64)
    cdef double[:, :] force_view = force

    cdef int i, j, dim, i_n_neigh, neigh
    cdef double strain, l, force_norm
    cdef double[3] f

    for i in range(nnodes):
        i_n_neigh = n_neigh[i]
        for neigh in range(i_n_neigh):
            j = nlist[i, neigh]

            if i < j:
                # Calculate total force
                l = ceuclid(r[i], r[j])
                strain = cstrain2(l, r0[i], r0[j])
                force_norm = strain * bond_stiffness

                # Calculate component of force in each dimension
                force_norm = force_norm / l
                for dim in range(3):
                    f[dim] = force_norm * (r[j, dim] - r[i, dim])

                # Add force to particle i, using Newton's third law subtract
                # force from j
                for dim in range(3):
                    force_view[i, dim] = force_view[i, dim] + f[dim]
                    force_view[j, dim] = force_view[j, dim] - f[dim]

        # Scale force by node volume
        for dim in range(3):
            force_view[i, dim] = force_view[i, dim] * volume[i]

    return force

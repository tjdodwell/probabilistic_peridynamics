import cython
import numpy as np
from libc.math cimport sqrt, abs


def euclid(r1, r2):
    return ceuclid(r1, r2)


cdef inline double ceuclid(double[:] r1, double[:] r2):
    cdef int imax = 3
    cdef double[3] dr

    for i in range(imax):
        dr[i] = r2[i] - r1[i]
        dr[i] = dr[i] * dr[i]

    return sqrt(dr[0] + dr[1] + dr[2])


def strain(r1, r2, r10, r20):
    return cstrain(r1, r2, r10, r20)


cdef inline double cstrain(double[:] r1, double[:] r2,
                           double[:] r10, double[:] r20):
    cdef double l, dl

    l = ceuclid(r1, r2)
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
    cdef int size = nlist.shape[1]

    cdef int[:, :] nlist_view = nlist
    cdef int[:] n_neigh_view = n_neigh

    cdef int i, j, i_n_neigh, neigh

    # Check neighbours for each node
    for i in range(nnodes):
        # Get current number of neighbours
        i_n_neigh = n_neigh_view[i]

        neigh = 0
        while neigh < i_n_neigh:
            j = nlist_view[i, neigh]

            if abs(cstrain(r[i], r[j], r0[i], r0[j])) < critical_strain:
                # Move onto the next neighbour
                neigh += 1
            else:
                # Remove this neighbour by replacing it with the last neighbour
                # on the list, then reducing the number of neighbours by 1
                nlist_view[i, neigh] = nlist_view[i, i_n_neigh-1]
                i_n_neigh -= 1

        n_neigh_view[i] = i_n_neigh


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

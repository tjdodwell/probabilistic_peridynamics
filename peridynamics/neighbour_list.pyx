import cython
from cython.parallel import prange
import numpy as np
from scipy.spatial.distance import euclidean
from libc.math cimport sqrt


def euclid(r1, r2):
    return ceuclid(r1, r2)


cdef inline double ceuclid(double[:] r1, double[:] r2) nogil:
    cdef int imax = 3
    cdef double[3] dr

    for i in range(imax):
        dr[i] = r2[i] - r1[i]
        dr[i] = dr[i] * dr[i]

    return sqrt(dr[0] + dr[1] + dr[2])


def family(double[:, :] r, double horizon):
    cdef int nnodes = r.shape[0]

    result = np.empty(nnodes, dtype=np.intc)
    cdef int[:] result_view = result

    cdef int tmp
    cdef int i, j

    for i in range(nnodes):
        tmp = 0
        for j in range(nnodes):
            if i == j:
                continue
            if ceuclid(r[i], r[j]) < horizon:
                tmp = tmp + 1

        result_view[i] = tmp

    return result


def create_neighbour_list(double[:, :] r, double horizon, int size):
    cdef int nnodes = r.shape[0]

    result = np.zeros((nnodes, size), dtype=np.intc)
    cdef int[:, :] result_view = result
    n_neigh = np.zeros(nnodes, dtype=np.intc)
    cdef int[:] n_neigh_view = n_neigh

    for i in range(nnodes):
        n_neigh_view[i] = 0
        for j in range(nnodes):
            if i != j:
                if ceuclid(r[i], r[j]) < horizon:
                    result_view[i, n_neigh_view[i]] = j
                    n_neigh_view[i] = n_neigh_view[i] + 1

    return result, n_neigh

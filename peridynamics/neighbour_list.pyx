from .spatial cimport ceuclid, cstrain
from libc.math cimport abs
import numpy as np


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


def create_crack(int[:, :] crack, int[:, :] nlist, int[:] n_neigh):
    cdef int n = crack.shape[0]

    cdef int icrack, i, j, neigh

    for icrack in range(n):
        i = crack[icrack][0]
        j = crack[icrack][1]

        # Iterate through i's neighbour list until j is found
        for neigh in range(n_neigh[i]):
            if nlist[i][neigh] == j:
                # Remove this neighbour by replacing it with the last neighbour
                # on the list, then reducing the number of neighbours by 1
                nlist[i, neigh] = nlist[i, n_neigh[i]-1]
                n_neigh[i] = n_neigh[i] - 1
                break

        # Iterate through j's neighbour list until i is found
        for neigh in range(n_neigh[j]):
            if nlist[j][neigh] == i:
                # Remove this neighbour by replacing it with the last neighbour
                # on the list, then reducing the number of neighbours by 1
                nlist[j, neigh] = nlist[j, n_neigh[j]-1]
                n_neigh[j] = n_neigh[j] - 1
                break

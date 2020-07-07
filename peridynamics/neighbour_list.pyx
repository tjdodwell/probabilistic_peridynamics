from .spatial cimport ceuclid, cstrain
from libc.math cimport abs
import numpy as np


def family(double[:, :] r, double horizon):
    """
    Determine the number of nodes within the horizon distance of each node.
    This is the total number of bonded nodes.

    :arg r: The coordinates of all nodes.
    :type r: :class:`numpy.ndarray`
    :arg float horizon: The horizon distance.

    :return: An array of the number of nodes within the horizon of each node.
    :rtype: :class:`numpy.ndarray`
    """
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
    """
    Build a neighbour list.

    :arg r: The coordinates of all nodes.
    :type r: :class:`numpy.ndarray`
    :arg float horizon: The horizon distance.
    :arg int size: The size of each row of the neighbour list. This is the
        maximum number of neighbours and should be equal to the maximum of
        of :func:`peridynamics.neighbour_list.family`.

    :return: A tuple of the neighbour list and number of neighbours for each
        node.
    :rtype: tuple(:class:`numpy.ndarray`, :class:`numpy.ndarray`)
    """
    cdef int nnodes = r.shape[0]

    result = -1.*np.ones((nnodes, size), dtype=np.intc)
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
    print(result)
    print(n_neigh)
    return result, n_neigh


def break_bonds(double[:, :] r, double[:, :]r0, int[:, :] nlist,
                int[:] n_neigh, double critical_strain):
    """
    Update the neighbour list and number of neighbours by breaking bonds which
    have exceeded the critical strain.

    :arg r: The current coordinates of each node.
    :type r: :class:`numpy.ndarray`
    :arg r0: The initial coordinates of each node.
    :type r0: :class:`numpy.ndarray`
    :arg nlist: The neighbour list
    :type nlist: :class:`numpy.ndarray`
    :arg n_neigh: The number of neighbours for each node.
    :type n_neigh: :class:`numpy.ndarray`
    :arg float critical_strain: The critical strain.
    """
    cdef int nnodes = nlist.shape[0]

    cdef int i, j, i_n_neigh, neigh
    cdef int j_n_neigh, jneigh

    # Check neighbours for each node
    for i in range(nnodes):
        # Get current number of neighbours
        i_n_neigh = n_neigh[i]

        neigh = 0
        while neigh < i_n_neigh:
            j = nlist[i, neigh]

            if i < j:
                if abs(cstrain(r[i], r[j], r0[i], r0[j])) < critical_strain:
                    # Move onto the next neighbour
                    neigh += 1
                else:
                    # Remove this neighbour by replacing it with the last
                    # neighbour on the list, then reducing the number of
                    # neighbours by 1.
                    # As neighbour `neigh` is now a new neighbour, we do not
                    # adcave the neighbour index
                    nlist[i, neigh] = nlist[i, i_n_neigh-1]
                    i_n_neigh -= 1

                    # Remove from j
                    j_n_neigh = n_neigh[j]
                    for jneigh in range(j_n_neigh):
                        if nlist[j, jneigh] == i:
                            nlist[j, jneigh] = nlist[j, j_n_neigh-1]
                            n_neigh[j] = n_neigh[j] - 1
                            break
            else:
                # Move onto the next neighbour
                neigh += 1

        n_neigh[i] = i_n_neigh


def create_crack(int[:, :] crack, int[:, :] nlist, int[:] n_neigh):
    """
    Create a crack by removing selected pairs from the neighbour list.

    :arg crack: An array giving the pairs between which to create the crack.
        Each row of this array should be the index of two nodes.
    :type crack: :class:`numpy.ndarray`
    :arg nlist: The neighbour list
    :type nlist: :class:`numpy.ndarray`
    :arg n_neigh: The number of neighbours for each node.
    :type n_neigh: :class:`numpy.ndarray`
    """
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

from .spatial cimport ceuclid, cstrain2
import numpy as np


def damage(int[:] n_neigh, int[:] family):
    """
    Calculate the damage for each node.

    Damage is defined as the ratio of broken bonds at a node to the total
    number of bonds at that node.

    :arg n_neigh: The current number of neighbours for each node, *i.e.* the
        number of unbroken bonds. dtype=numpy.int32.
    :type n_neigh: :class:`numpy.ndarray`
    :arg family: The total (initial) number of bonds for each node.
        dtype=numpy.int32.
    :type family: :class:`numpy.ndarray`
    """
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
    """
    Calculate the force due to bonds on each node.

    :arg r: The current coordinates of each node.
    :type r: :class:`numpy.ndarray`
    :arg r0: The initial coordinates of each node.
    :type r0: :class:`numpy.ndarray`
    :arg nlist: The neighbour list
    :type nlist: :class:`numpy.ndarray`
    :arg n_neigh: The number of neighbours for each node.
    :type n_neigh: :class:`numpy.ndarray`
    :arg volume: The volume of each node.
    :type volume: :class:`numpy.ndarray`
    :arg float bond_stiffness: The bond stiffness.
    """
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

from .spatial cimport ceuclid
import numpy as np


def set_imprecise_stiffness_correction(
        double[:, :] stiffness_corrections, int[:, :] nlist, int[:] n_neigh,
        double average_volume, double family_volume_bulk):
    """
    Calculate the stiffness corrections using an average nodal volume.

    :arg stiffness_corrections: The stiffness corrections.
    :type stiffness_corrections: :class:`numpy.ndarray`
    :arg nlist: The neighbour list
    :type nlist: :class:`numpy.ndarray`
    :arg n_neigh: The number of neighbours for each node.
    :type n_neigh: :class:`numpy.ndarray`
    :arg double average_volume: The average nodal volume.
    :arg double family_volume_bulk: Volume of a family in the bulk material.
    """
    cdef int nnodes = nlist.shape[0]
    cdef double[:] family_volumes = np.zeros(nnodes, dtype=np.float64)
    cdef double family_volume_i, family_volume_j
    cdef double correction

    cdef int i, j, i_n_neigh, neigh
    cdef int j_n_neigh, jneigh
    
    for i in range(nnodes):
        # Get current number of neighbours
        i_n_neigh = n_neigh[i]
        family_volume_i = i_n_neigh * average_volume

        neigh = 0
        while neigh < i_n_neigh:
            j = nlist[i, neigh]

            if i < j:
                j_n_neigh = n_neigh[j]
                family_volume_j = j_n_neigh * average_volume
                correction = 2. * family_volume_bulk / (
                    family_volume_i + family_volume_j)
                # Set stiffness correction factor for this bond
                stiffness_corrections[i, neigh] *= correction

                # Also set for j, since it is symmetric
                for jneigh in range(j_n_neigh):
                    if nlist[j, jneigh] == i:
                        stiffness_corrections[j, jneigh] *= correction
                        break

            # Move on to the next neighbour
            neigh += 1


def set_precise_stiffness_correction(
        double[:, :] stiffness_corrections, int[:, :] nlist, int[:] n_neigh,
        double[:] volume, double family_volume_bulk):
    """
    Calculate the stiffness corrections given actual nodal volumes.

    :arg stiffness_corrections: The stiffness corrections.
    :type stiffness_corrections: :class:`numpy.ndarray`
    :arg nlist: The neighbour list
    :type nlist: :class:`numpy.ndarray`
    :arg n_neigh: The number of neighbours for each node.
    :type n_neigh: :class:`numpy.ndarray`
    :arg volume: The nodal volumes.
    :type volume: :class:`numpy.ndarray`
    :arg double family_volume_bulk: Volume of a family in the bulk material.
    """
    cdef int nnodes = nlist.shape[0]
    cdef double[:] family_volumes = np.zeros(nnodes, dtype=np.float64)
    cdef double family_volume_i, family_volume_j
    cdef double tmp
    cdef double correction

    cdef int i, j, i_n_neigh, neigh
    cdef int j_n_neigh, jneigh

    # Calculate the family volumes
    for i in range(nnodes):
        tmp = 0.0
        i_n_neigh = n_neigh[i]
        for neigh in range(i_n_neigh):
            j = nlist[i, neigh]
            tmp += volume[j]
        family_volumes[i] = tmp

    for i in range(nnodes):
        # Get current number of neighbours
        i_n_neigh = n_neigh[i]
        family_volume_i = family_volumes[i]

        neigh = 0
        while neigh < i_n_neigh:
            j = nlist[i, neigh]

            if i < j:
                family_volume_j = family_volumes[j]
                correction = 2. * family_volume_bulk / (
                    family_volume_i + family_volume_j)
                # Set stiffness correction factor for this bond
                stiffness_corrections[i, neigh] *= correction
        
                # Also set for j, since it is symmetric
                j_n_neigh = n_neigh[j]
                for jneigh in range(j_n_neigh):
                    if nlist[j, jneigh] == i:
                        stiffness_corrections[j, jneigh] *= correction
                        break

            # Move on to the next neighbour
            neigh += 1


def set_volume_correction(double[:, :]volume_corrections, double[:, :]r0,
                      int[:, :] nlist, int[:] n_neigh, double horizon,
                      double node_radius, int volume_correction):
    """
    Calculate the partial volume corrections given the initial coordinates,
    peridynamic horizon and node radius.

    :arg volume_corrections: The volume corrections.
    :type volume_corrections: :class:`numpy.ndarray`
    :arg r0: The initial coordinates of each node.
    :type r0: :class:`numpy.ndarray`
    :arg nlist: The neighbour list
    :type nlist: :class:`numpy.ndarray`
    :arg n_neigh: The number of neighbours for each node.
    :type n_neigh: :class:`numpy.ndarray`
    :arg float horizon: The critical strain.
    :arg float node_radius: The node radius.
    :arg int volume_correction: A flag variable denoting the algorithm used.
    """
    cdef int nnodes = nlist.shape[0]

    cdef int i, j, i_n_neigh, neigh
    cdef int j_n_neigh, jneigh

    cdef double correction

    # Check neighbours for each node
    for i in range(nnodes):
        # Get current number of neighbours
        i_n_neigh = n_neigh[i]

        neigh = 0
        while neigh < i_n_neigh:
            j = nlist[i, neigh]

            if i < j:
                # Calculate the correction depending on the algorithm
                if volume_correction == 0:
                    correction = cvolume_correction(
                        r0[i], r0[j], horizon, node_radius)
                else:
                    # USERNOTE: Place your own volume correction algorithm here
                    correction = 1.00
                # Set volume correction for this bond
                volume_corrections[i, neigh] = correction

                # Also set for j, since it is symmetric
                j_n_neigh = n_neigh[j]
                for jneigh in range(j_n_neigh):
                    if nlist[j, jneigh] == i:
                        volume_corrections[j, jneigh] = correction
                        break
            # Move on to the next neighbour
            neigh += 1


cdef inline double cvolume_correction(double[:] r10, double[:] r20,
                                      double horizon, double node_radius):
    """
    C function for calculating the partial volume correction given the initial
    coordinates, peridynamic horizon and node radius.
    """

    l0 = ceuclid(r10, r20)

    if (l0 <= horizon - node_radius):
        return 1.00
    elif (l0 <= horizon + node_radius):
        return (horizon + node_radius - l0) / (2.0 * node_radius)
    else:
        return 0.00


def set_micromodulus_function(
        double[:, :]micromodulus_values, double[:, :]r0, int[:, :] nlist,
        int[:] n_neigh, double horizon, double node_radius,
        int micromodulus_function):
    """
    Calculate the normalised conical micromodulus function values given the
    initial coordinates and peridynamic horizon.

    :arg micromodulus_values: The micromodulus values.
    :type micromodulus_values: :class:`numpy.ndarray`
    :arg r0: The initial coordinates of each node.
    :type r0: :class:`numpy.ndarray`
    :arg nlist: The neighbour list
    :type nlist: :class:`numpy.ndarray`
    :arg n_neigh: The number of neighbours for each node.
    :type n_neigh: :class:`numpy.ndarray`
    :arg float horizon: The critical strain.
    """
    cdef int nnodes = nlist.shape[0]

    cdef int i, j, i_n_neigh, neigh
    cdef int j_n_neigh, jneigh

    cdef double value

    # Check neighbours for each node
    for i in range(nnodes):
        # Get current number of neighbours
        i_n_neigh = n_neigh[i]

        neigh = 0
        while neigh < i_n_neigh:
            j = nlist[i, neigh]

            if i < j:
                # Calculate the correction depending on the algorithm
                if micromodulus_function == 0:
                    value = cmicromodulus_connical(
                        r0[i], r0[j], horizon)
                else:
                    # USERNOTE: Place your own micromodulus function here
                    value = 1.00
                # Set volume correction for this bond
                micromodulus_values[i, neigh] = value

                # Also set for j, since it is symmetric
                j_n_neigh = n_neigh[j]
                for jneigh in range(j_n_neigh):
                    if nlist[j, jneigh] == i:
                        micromodulus_values[j, jneigh] = value
                        break
            # Move on to the next neighbour
            neigh += 1


cdef inline double cmicromodulus_connical(double[:] r10, double[:] r20,
                                      double horizon):
    """
    C function for calculating the normalised connical micromodulus function
    given the initial coordinates and peridynamic horizon.
    """

    l0 = ceuclid(r10, r20)

    if (l0 <= horizon):
        return (horizon - l0) / (horizon)
    else:
        return 0.00

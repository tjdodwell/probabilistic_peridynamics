def create_crack_cython(int[:, :] crack, int[:, :] nlist, int[:] n_neigh):
    """
    Create a crack by removing selected pairs from the neighbour list.

    This method is special to the cython integrators, such as
    :class:`peridynamics.integrators.Euler`. The key difference between this
    method and :meth:`peridynamics.create_crack.create_crack_cl` is that it
    will remove the broken bond by replacing a neighbour with the last
    neighbour on the list, as opposed to replacing it with a flag of -1.

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


def create_crack_cl(int[:, :] crack, int[:, :] nlist, int[:] n_neigh,
                    int[:] family):
    """
    Create a crack by removing selected pairs from the neighbour list.

    This method is special to the OpenCL integrators, such as
    :class:`peridynamics.integrators.EulerCL`. The key difference between this
    method and :meth:`peridynamics.create_crack.create_crack_cython` is that it
    will remove the broken bond by replacing a neighbour with a flag of -1, as
    opposed to replacing it with the last neighbour on the list. This is due to
    the slightly different data structure of nlist for the OpenCL integrators.

    :arg crack: An array giving the pairs between which to create the crack.
        Each row of this array should be the index of two nodes.
    :type crack: :class:`numpy.ndarray`
    :arg nlist: The neighbour list
    :type nlist: :class:`numpy.ndarray`
    :arg n_neigh: The number of neighbours for each node.
    :type n_neigh: :class:`numpy.ndarray`
    :arg family: The initial number of neighbours for each node.
    :type family: :class:`numpy.ndarray`
    """
    cdef int n = crack.shape[0]

    cdef int icrack, i, j, neigh

    for icrack in range(n):
        i = crack[icrack][0]
        j = crack[icrack][1]

        # Iterate through i's neighbour list until j is found
        for neigh in range(family[i]):
            if nlist[i][neigh] == j:
                # Remove this neighbour by replacing it with -1, then reducing
                # the number of neighbours by 1
                nlist[i, neigh] = -1
                n_neigh[i] = n_neigh[i] - 1
                break

        # Iterate through j's neighbour list until i is found
        for neigh in range(family[j]):
            if nlist[j][neigh] == i:
                # Remove this neighbour by replacing it with -1, then reducing
                # the number of neighbours by 1
                nlist[j, neigh] = -1
                n_neigh[j] = n_neigh[j] - 1
                break

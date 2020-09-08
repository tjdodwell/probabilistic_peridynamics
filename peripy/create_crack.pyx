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
                # on the list, remove the last neighbour on the list by
                # replacing it with -1, then reducing the number of neighbours
                # by 1
                nlist[i, neigh] = nlist[i, n_neigh[i]-1]
                nlist[i, n_neigh[i]-1] = -1
                n_neigh[i] = n_neigh[i] - 1
                break

        # Iterate through j's neighbour list until i is found
        for neigh in range(n_neigh[j]):
            if nlist[j][neigh] == i:
                # Remove this neighbour by replacing it with the last neighbour
                # on the list, remove the last neighbour on the list by
                # replacing it with -1, then reducing the number of neighbours
                # by 1
                nlist[j, neigh] = nlist[j, n_neigh[j]-1]
                nlist[j, n_neigh[j]-1] = -1
                n_neigh[j] = n_neigh[j] - 1
                break

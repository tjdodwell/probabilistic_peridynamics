#pragma OPENCL EXTENSION cl_khr_fp64 : enable


__kernel void break_bonds(__global const double* r, global const double* r0,
                          __global int* nlist, __global int* n_neigh,
                          int max_neigh, double critical_strain) {
    /* Update the neighbour list by breaking any bonds which have exceeded the
     * critical strain.
     * r - An (n,3) array of current coordinates.
     * r0 - An (n,3) array of initial coordinates.
     * nlist - An (n,max_neigh) array representing the neighbours of each node.
     * n_niegh - An (n) array of the number of neighbours of each node.
     * max_neigh - The maximum number of neighbours.
     * critical_strain - The critical strain, at and above which bonds will be
     *     broken. */
    int i = get_global_id(0);

    int i_n_neigh = n_neigh[i];

    int neigh = 0;
    while (neigh < i_n_neigh) {
        int j = nlist[i*max_neigh + neigh];

        if (fabs(strain2(r, r0, i, j)) < critical_strain) {
            neigh += 1;
        } else {
            nlist[i*max_neigh + neigh] = nlist[i*max_neigh + i_n_neigh-1];
            i_n_neigh -= 1;
        }
    }

    n_neigh[i] = i_n_neigh;
}

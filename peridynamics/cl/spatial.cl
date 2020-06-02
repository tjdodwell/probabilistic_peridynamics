#pragma OPENCL EXTENSION cl_khr_fp64 : enable


double euclid(__global const double* r, int i, int j) {
    /* Calculate the Euclidean distance between two points.
     *
     * r - An (n,3) array of coordinates.
     * i - The index of the first point.
     * j - The index of the second point.
     *
     * returns - The Euclidean distance between i and j. */
    int il = i*3;
    int jl = j*3;

    double dx = r[jl] - r[il];
    double dy = r[jl+1] - r[il+1];
    double dz = r[jl+2] - r[il+2];

    return sqrt(dx*dx + dy*dy + dz*dz);
}


double strain(__global const double* r0, int i, int j, double l) {
    /* Calculate the strain between two points given their initial coordinates
     * and current distance.
     *
     *  r0 - An (n,3) array of initial coordinates.
     * i - The index of the first point.
     * j - The index of the second point.
     * l - The current distance between i and j.
     *
     * returns - The strain between i and j. */
    double l0 = euclid(r0, i, j);
    double dl = l - l0;

    return dl / l0;
}


double strain2(__global const double* r, __global const double* r0, int i,
               int j) {
    /* Calculate the strain between two points given their initial and current
     * coordinates.
     *
     *  r0 - An (n,3) array of initial coordinates.
     * i - The index of the first point.
     * j - The index of the second point.
     * l - The current distance between i and j.
     *
     * returns - The strain between i and j. */
    double l = euclid(r, i, j);
    double l0 = euclid(r0, i, j);
    double dl = l - l0;

    return dl / l0;
}

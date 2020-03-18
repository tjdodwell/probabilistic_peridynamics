#pragma OPENCL EXTENSION cl_khr_fp64 : enable


double euclid(__global const double* r, int i, int j) {
    int il = i*3;
    int jl = j*3;

    double dx = r[jl] - r[il];
    double dy = r[jl+1] - r[il+1];
    double dz = r[jl+2] - r[il+2];

    return sqrt(dx*dx + dy*dy + dz*dz);
}


__kernel void dist(__global const double* r, __global double* d) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    int n = get_global_size(0);

    d[i*n + j] = euclid(r, i, j);
}


__kernel void neighbourhood(__global const double* r, double threshold,
                            __global bool* nhood) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    int n = get_global_size(0);

    int index = i*n + j;

    if (i == j) {
        nhood[index] = false;
    } else {
        nhood[i*n + j] = euclid(r, i, j) < threshold;
    }
}


__kernel void strain(__global const double* r, __global const double* d0,
                     __global double* strain) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    int n = get_global_size(0);

    int index = i*n +j;

    double l0 = d0[index];
    if (l0 == 0.) {
        strain[index] = 0.;
    } else {
        double l = euclid(r, i, j);
        strain[index] = (l - l0) / l0;
    }
}

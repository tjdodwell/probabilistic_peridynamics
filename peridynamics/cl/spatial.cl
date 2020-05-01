#pragma OPENCL EXTENSION cl_khr_fp64 : enable


double euclid(__global const double* r, int i, int j) {
    int il = i*3;
    int jl = j*3;

    double dx = r[jl] - r[il];
    double dy = r[jl+1] - r[il+1];
    double dz = r[jl+2] - r[il+2];

    return sqrt(dx*dx + dy*dy + dz*dz);
}


double strain(__global const double* r0, int i, int j, double l) {
    int il = i*3;
    int jl = j*3;

    double dx = r0[jl] - r0[il];
    double dy = r0[jl+1] - r0[il+1];
    double dz = r0[jl+2] - r0[il+2];

    double l0 = sqrt(dx*dx + dy*dy + dz*dz);
    double dl = l - l0;

    return dl / l0;
}


double strain2(__global const double* r, __global const double* r0, int i,
               int j) {
    double l = euclid(r, i, j);
    double l0 = euclid(r0, i, j);
    double dl = l - l0;

    return dl / l0;
}

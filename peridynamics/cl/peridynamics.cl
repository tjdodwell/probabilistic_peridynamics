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


__kernel void neighbourhood(__global const double* r, double threshold,
                            __global bool* nhood) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    int n = get_global_size(1);

    int index = i*n + j;

    if (i == j) {
        nhood[index] = false;
    } else {
        nhood[i*n + j] = euclid(r, i, j) < threshold;
    }
}


__kernel void dist(__global const double* r, __global const bool* nhood,
                   __global double* d) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    int n = get_global_size(1);

    int index = i*n + j;
    if (nhood[index]) {
        d[index] = euclid(r, i, j);
    }
}


__kernel void break_bonds(__global const double* strain, double critical_strain,
                          __global bool* nhood) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    int n = get_global_size(1);

    int index = i*n +j;

    if (nhood[index]) {
        if (fabs(strain[index]) > critical_strain) {
            nhood[index] = false;
        }
    }
}


__kernel void damage(__global const int* n_neigh, __global const int* family,
                     __global double* damage){
    int i = get_global_id(0);

    int ifamily = family[i];
    damage[i] = (double)(ifamily - n_neigh[i])/ifamily;
}


__kernel void bond_force(__global const double* r, __global const double* r0,
                         __global const int* nlist, global const int* n_neigh,
                         int max_neigh, __global const double* volume,
                         double bond_stiffness, __global double* f) {
    int i = get_global_id(0);

    double fi[3] = {0.0, 0.0, 0.0};

    for(int neigh=0; neigh<n_neigh[i]; neigh++) {
        int j = nlist[i*max_neigh + neigh];

        double l = euclid(r, i, j);

        double force_norm = strain(r0, i, j, l) * bond_stiffness;
        force_norm = force_norm / l;

        #pragma unroll
        for(int dim=0; dim<3; dim++) {
            fi[dim] += force_norm * (r[j*3 + dim] - r[i*3 + dim]);
        }
    }

    #pragma unroll
    for(int dim=0; dim<3; dim++) {
        fi[dim] *= volume[i];
    }

    #pragma unroll
    for(int dim=0; dim<3; dim++) {
        f[i*3 + dim] = fi[dim];
    }
}

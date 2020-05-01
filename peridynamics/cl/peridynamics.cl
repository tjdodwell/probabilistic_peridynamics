#pragma OPENCL EXTENSION cl_khr_fp64 : enable


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

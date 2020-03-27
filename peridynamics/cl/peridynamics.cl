#pragma OPENCL EXTENSION cl_khr_fp64 : enable


double euclid(__global const double* r, int i, int j) {
    int il = i*3;
    int jl = j*3;

    double dx = r[jl] - r[il];
    double dy = r[jl+1] - r[il+1];
    double dz = r[jl+2] - r[il+2];

    return sqrt(dx*dx + dy*dy + dz*dz);
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


__kernel void dist(__global const double* r, __global const bool* nhood,
                   __global double* d) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    int n = get_global_size(0);

    int index = i*n + j;
    if (nhood[index]) {
        d[index] = euclid(r, i, j);
    }
}


__kernel void strain(__global const double* r, __global const double* d0,
                     __global const bool* nhood, __global double* strain) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    int n = get_global_size(0);

    int index = i*n +j;

    if (nhood[index]) {
        double l0 = d0[index];
        if (l0 == 0.) {
            strain[index] = 0.;
        } else {
            double l = euclid(r, i, j);
            strain[index] = (l - l0) / l0;
        }
    }
}


__kernel void break_bonds(__global const double* strain, double critical_strain,
                          __global bool* nhood) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    int n = get_global_size(0);

    int index = i*n +j;

    if (nhood[index]) {
        if (fabs(strain[index]) > critical_strain) {
            nhood[index] = false;
        }
    }
}


__kernel void damage1(__global const bool* nhood, __local int* b,
                      __global int* partials) {
    int gid0 = get_global_id(0);
    int gid1 = get_global_id(1);
    int gsize1 = get_global_size(1);

    int lid = get_local_id(0);
    int lsize = get_local_size(0);
    int wg = get_group_id(0);

    // Copy to local memory, cast true to 1 and false to 0
    b[lid] = (int)nhood[gid0*gsize1 + gid1];
    barrier(CLK_LOCAL_MEM_FENCE);

    // Reduction within work group, sum is left in b[0]
    for (int stride=lsize>>1; stride>0; stride>>=1) {
        if (lid < stride) {
            b[lid] += b[lid+stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Local thread 0 copies its work group sum to the result array
    if (lid == 0) {
        // row is wg
        // col is gid1
        partials[wg*gsize1 + gid1] = b[0];
    }
}


__kernel void damage2(__global int* partials, int n_partials,
                      __global const int* family, __global double* damage) {
    int i = get_global_id(0);
    int n = get_global_size(0);

    // Complete summing the number of unbroken bonds
    damage[i] = 0.0;
    for (int j=0; j<n_partials; j++) {
        damage[i] += (double)partials[j*n+i];
    }

    // Damage calculation
    // (initially_unbroken - current_unbroken)/initially_unbroken
    // == 1 - current_unbroken/initially_unbroken
    damage[i] = 1.0 - damage[i]/family[i];
}

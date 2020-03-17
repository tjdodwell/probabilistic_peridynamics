float euclid(__global const float* r, int i, int j) {
    int il = i*3;
    int jl = j*3;

    float dx = r[jl] - r[il];
    float dy = r[jl+1] - r[il+1];
    float dz = r[jl+2] - r[il+2];

    return sqrt(dx*dx + dy*dy + dz*dz);
}

__kernel void dist(__global const float* r, __global float* d) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    int n = get_global_size(0);

    d[i*n + j] = euclid(r, i, j);
}

__kernel void strain(__global const float* r, __global const float* d0,
                     __global float* strain) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    int n = get_global_size(0);

    int index = i*n +j;

    float l0 = d0[index];
    if (l0 == 0.) {
        strain[index] = 0.;
    } else {
        float l = euclid(r, i, j);
        strain[index] = (l - l0) / l0;
    }
}

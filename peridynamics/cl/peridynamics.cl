// float euclid(__global const float* r, int i, int j) {
//     float dx = r[i

__kernel void dist(__global const float* r, __global float* d) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    int n = get_global_size(0);

    int il = i*3;
    int jl = j*3;

    float dx = r[jl] - r[il];
    float dy = r[jl+1] - r[il+1];
    float dz = r[jl+2] - r[il+2];

    d[i*n + j] = sqrt(dx*dx + dy*dy + dz*dz);
}

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

// Constants
#define DPN 3
// MAX_HORIZON_LENGTH, PD_R, PD_DX, PD_DT, PD_NODE_NO, PD_DPN_NODE_NO, PD_REGIME_NO will be defined on JIT compiler's command line

__kernel void
	update_displacement(
    	__global double const* ud,
    	__global double* u,
		__global int const* BC_types,
		__global double const* BC_values,
		double BC_scale
	){
    /* Calculate the displacement of each node.
     *
     * ud - An (n,3) array of the velocities of each node.
     * u - An (n,3) array of the current displacements of each node.
     * BC_types - An (n,3) array of the boundary condition types...
     * a value of 2 denotes an unconstrained node.
     * BC_values - An (n,3) array of the boundary condition values applied nodes.
     * BC_scale - The scalar value applied to the displacement BCs. */
	const int i = get_global_id(0);

	if (i < PD_DPN_NODE_NO)	{
		u[i] = (BC_types[i] == 2 ? (u[i] + PD_DT * ud[i]) : (u[i] + BC_scale * BC_values[i]));
	}
}

__kernel void
	bond_force(
    __global double const* u,
    __global double* ud,
    __global double const* r0,
    __global double const* vols,
	__global int* nlist,
    __global int* n_neigh,
    __global int const* fc_types,
    __global double const* fc_values,
    __local double* local_cache_x,
    __local double* local_cache_y,
    __local double* local_cache_z,
    double force_load_scale,
    double stiffness,
    double critical_strain
	) {
    /* Calculate the force due to bonds on each node and update node velocities.
     *
     * u - An (n,3) array of the current displacements of the particles.
     * ud - An (n,3) array of the current velocities of the particles.
     * vols - the volumes of each of the nodes.
     * nlist - An (n, local_size) array containing the neighbour lists,
     *     a value of -1 corresponds to a broken bond.
     * r0 - An (n,3) array of the coordinates of the nodes in the initial state.
     *     stiffness_corrections - An (n * max_neigh) array of bond stiffness correction factors.
     *     critical_stretches - An (n * max_neigh) array of bond critical strains.
     * fc_types - An (n,3) array of force boundary condition types...
     *     a value of 2 denotes a particle that is not externally loaded.
     * fc_values - An (n,3) array of the force boundary condition values applied to particles.
     * local_cache_x - local (local_size) array to store the x components of the bond forces.
     * local_cache_y - local (local_size) array to store the y components of the bond forces.
     * local_cache_z - local (local_size) array to store the z components of the bond forces.
     * force_load_scale - scale factor applied to 
     * bond_stiffness - The bond stiffness.
     * critical_strain - The critical strain, at and above which bonds will be
     *     broken. */
    // global_id is the bond number
    const int global_id = get_global_id(0);
    // local_id is the LOCAL node id in range [0, max_neigh] of a node in this parent node's family
	const int local_id = get_local_id(0);
    // local_size is the max_neigh, usually 128 or 256 depending on the problem
    const int local_size = get_local_size(0);

	if ((global_id < (PD_NODE_NO * local_size)) && (local_id >= 0) && (local_id < local_size)) {   
        // Find corresponding node id
        const double temp = global_id / local_size;
        const int node_id_i = floor(temp);

        // Access local node within node_id_i's horizon with corresponding node_id_j,
        const int node_id_j = nlist[global_id];

        // If bond is not broken
        if (node_id_j != -1) {
        const double xi_x = r0[DPN * node_id_j + 0] - r0[DPN * node_id_i + 0];
        const double xi_y = r0[DPN * node_id_j + 1] - r0[DPN * node_id_i + 1];
        const double xi_z = r0[DPN * node_id_j + 2] - r0[DPN * node_id_i + 2];

        const double xi_eta_x = u[DPN * node_id_j + 0] - u[DPN * node_id_i + 0] + xi_x;
        const double xi_eta_y = u[DPN * node_id_j + 1] - u[DPN * node_id_i + 1] + xi_y;
        const double xi_eta_z = u[DPN * node_id_j + 2] - u[DPN * node_id_i + 2] + xi_z;

        const double xi = sqrt(xi_x * xi_x + xi_y * xi_y + xi_z * xi_z);
        const double y = sqrt(xi_eta_x * xi_eta_x + xi_eta_y * xi_eta_y + xi_eta_z * xi_eta_z);
        const double y_xi = (y - xi);

        const double cx = xi_eta_x / y;
        const double cy = xi_eta_y / y;
        const double cz = xi_eta_z / y;

        const double _E = stiffness;
        const double _A = vols[node_id_j];
        const double _L = xi;

        const double _EAL = _E * _A / _L;

        // Copy bond forces into local memory
        local_cache_x[local_id] = _EAL * cx * y_xi;
        local_cache_y[local_id] = _EAL * cy * y_xi;
        local_cache_z[local_id] = _EAL * cz * y_xi;

        // Check for state of bonds here, and break it if necessary
        const double s0 = critical_strain;
        const double s = (y - xi) / xi;
        if (s > s0) {
            nlist[global_id] = -1;  // Break the bond
            n_neigh[node_id_i] -= 1;
        }
    }
    // bond is broken
    else {
        local_cache_x[local_id] = 0.00;
        local_cache_y[local_id] = 0.00;
        local_cache_z[local_id] = 0.00;
    }

    // Wait for all threads to catch up
    barrier(CLK_LOCAL_MEM_FENCE);
    // Parallel reduction of the bond force onto node force
    for (int i = local_size/2; i > 0; i /= 2) {
        if(local_id < i) {
            local_cache_x[local_id] += local_cache_x[local_id + i];
            local_cache_y[local_id] += local_cache_y[local_id + i];
            local_cache_z[local_id] += local_cache_z[local_id + i];
        } 
        //Wait for all threads to catch up 
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (!local_id) {
        //Get the reduced forces
        // node_no == node_id_i
        int node_no = global_id/local_size;
        // Update accelerations in each direction
        ud[DPN * node_no + 0] = (fc_types[DPN * node_no + 0] == 2 ? local_cache_x[0] : (local_cache_x[0] + force_load_scale * fc_values[DPN * node_no + 0]));
        ud[DPN * node_no + 1] = (fc_types[DPN * node_no + 1] == 2 ? local_cache_y[0] : (local_cache_y[0] + force_load_scale * fc_values[DPN * node_no + 1]));
        ud[DPN * node_no + 2] = (fc_types[DPN * node_no + 2] == 2 ? local_cache_z[0] : (local_cache_z[0] + force_load_scale * fc_values[DPN * node_no + 2]));
    }
  }
}

__kernel void 
    damage(
        __global int const* nlist,
		__global int const* family,
        __global double* damage,
        __local double* local_cache
    ) {
    /* Calculate the damage of each node.
     *
     * nlist - An (n, local_size) array containing the neighbour lists,
     *     a value of -1 corresponds to a broken bond.
     * family - An (n) array of the initial number of neighbours for each node.
     * damage - An (n) array of the damage for each node.
     * local_cache - A (local_size) local array for the parallel reduction of damage. */
    int global_id = get_global_id(0); 
    int local_id = get_local_id(0); 
    // local size is the max_neigh and must be a power of 2
    int local_size = get_local_size(0); 
    
    //Copy values into local memory 
    local_cache[local_id] = nlist[global_id] != -1 ? 1.00 : 0.00; 

    //Wait for all threads to catch up 
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int i = local_size/2; i > 0; i /= 2) {
        if(local_id < i) {
            local_cache[local_id] += local_cache[local_id + i];
        } 
        //Wait for all threads to catch up 
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (!local_id) {
        //Get the reduced forces
        int node_id = global_id/local_size;
        // Update damage
        damage[node_id] = 1.00 - (double) local_cache[0] / (double) (family[node_id]);
    }
}

__kernel void damage_new(__global const int* n_neigh, __global const int* family,
                     __global double* damage){
    /* Calculate the damage of each node.
     *
     * n_neigh - An (n) array of the number of neighbours (particles bound) for
     *     each node.
     * family - An (n) array of the initial number of neighbours for each node.
     * damage - An (n) array of the damage for each node. */
    int i = get_global_id(0);

    int ifamily = family[i];
    damage[i] = (double)(ifamily - n_neigh[i])/ifamily;
}

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

// Constants
// nnodes, dof, dof_nnodes will be defined on JIT compiler's command line

__kernel void
	update_displacement(
    	__global double const* ud,
    	__global double* u,
		__global int const* bc_types,
		__global double const* bc_values,
		double bc_scale,
        double dt
	){
    /* Calculate the displacement of each node.
     *
     * ud - An (n,3) array of the velocities of each node.
     * u - An (n,3) array of the current displacements of each node.
     * BC_types - An (n,3) array of the boundary condition types...
     * a value of 0 denotes an unconstrained node.
     * bc_values - An (n,3) array of the boundary condition values applied to the nodes.
     * bc_scale - The scalar value applied to the displacement BCs. */
	const int i = get_global_id(0);

	if (i < dof_nnodes)	{
		u[i] = (bc_types[i] == 0 ? (u[i] + dt * ud[i]) : (u[i] + bc_scale * bc_values[i]));
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
    double fc_scale,
    double bond_stiffness,
    double critical_stretch
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
     *     a value of 0 denotes a particle that is not externally loaded.
     * fc_values - An (n,3) array of the force boundary condition values applied to particles.
     * local_cache_x - local (local_size) array to store the x components of the bond forces.
     * local_cache_y - local (local_size) array to store the y components of the bond forces.
     * local_cache_z - local (local_size) array to store the z components of the bond forces.
     * fc_scale - scale factor applied to 
     * bond_stiffness - The bond stiffness.
     * critical_strain - The critical strain, at and above which bonds will be
     *     broken. */
    // global_id is the bond number
    const int global_id = get_global_id(0);
    // local_id is the LOCAL node id in range [0, max_neigh] of a node in this parent node's family
	const int local_id = get_local_id(0);
    // local_size is the max_neigh, usually 128 or 256 depending on the problem
    const int local_size = get_local_size(0);

	if ((global_id < (nnodes * local_size)) && (local_id >= 0) && (local_id < local_size)) {   
        // Find corresponding node id
        const double temp = global_id / local_size;
        const int node_id_i = floor(temp);

        // Access local node within node_id_i's horizon with corresponding node_id_j,
        const int node_id_j = nlist[global_id];

        // If bond is not broken
        if (node_id_j != -1) {
        const double xi_x = r0[dof * node_id_j + 0] - r0[dof * node_id_i + 0];
        const double xi_y = r0[dof * node_id_j + 1] - r0[dof * node_id_i + 1];
        const double xi_z = r0[dof * node_id_j + 2] - r0[dof * node_id_i + 2];

        const double xi_eta_x = u[dof * node_id_j + 0] - u[dof * node_id_i + 0] + xi_x;
        const double xi_eta_y = u[dof * node_id_j + 1] - u[dof * node_id_i + 1] + xi_y;
        const double xi_eta_z = u[dof * node_id_j + 2] - u[dof * node_id_i + 2] + xi_z;

        const double xi = sqrt(xi_x * xi_x + xi_y * xi_y + xi_z * xi_z);
        const double y = sqrt(xi_eta_x * xi_eta_x + xi_eta_y * xi_eta_y + xi_eta_z * xi_eta_z);
        const double s = (y -  xi)/ xi;

        const double cx = xi_eta_x / y;
        const double cy = xi_eta_y / y;
        const double cz = xi_eta_z / y;

        const double f = s * bond_stiffness * vols[node_id_j];

        // Copy bond forces into local memory
        local_cache_x[local_id] = f * cx;
        local_cache_y[local_id] = f * cy;
        local_cache_z[local_id] = f * cz;

        // Check for state of bonds here, and break it if necessary
        if (s > critical_stretch) {
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
        ud[dof * node_no + 0] = (fc_types[dof * node_no + 0] == 0 ? local_cache_x[0] : (local_cache_x[0] + fc_scale * fc_values[dof * node_no + 0]));
        ud[dof * node_no + 1] = (fc_types[dof * node_no + 1] == 0 ? local_cache_y[0] : (local_cache_y[0] + fc_scale * fc_values[dof * node_no + 1]));
        ud[dof * node_no + 2] = (fc_types[dof * node_no + 2] == 0 ? local_cache_z[0] : (local_cache_z[0] + fc_scale * fc_values[dof * node_no + 2]));
    }
  }
}
#pragma OPENCL EXTENSION cl_khr_fp64 : enable


__kernel void
	bond_force1(
    __global double const* u,
    __global double* force,
    __global double* body_force,
    __global double const* r0,
    __global double const* vols,
	__global int* nlist,
    __global int const* fc_types,
    __global double const* fc_values,
    __global double const* stiffness_corrections,
    __global int const* bond_types,
    __global int* regimes,
    __global float const* plus_cs,
    __local double* local_cache_x,
    __local double* local_cache_y,
    __local double* local_cache_z,
    double bond_stiffness,
    double critical_stretch,
    double fc_scale,
    int nregimes
	) {
    /* Calculate the force due to bonds on each node.
     *
     * This bond_force function is for the simple case of no stiffness corrections and no bond types.
     *
     * u - An (n,3) array of the current displacements of the particles.
     * force - An (n,3) array of the current forces on the particles.
     * body_force - An (n,3) array of the current internal body forces of the particles.
     * r0 - An (n,3) array of the coordinates of the nodes in the initial state.
     * vols - the volumes of each of the nodes.
     * nlist - An (n, local_size) array containing the neighbour lists,
     *     a value of -1 corresponds to a broken bond.
     * fc_types - An (n,3) array of force boundary condition types,
     *     a value of 0 denotes a particle that is not externally loaded.
     * fc_values - An (n,3) array of the force boundary condition values applied to particles.
     * stiffness_corrections - Not applied in this bond_force kernel. Placeholder argument.
     * bond_types - Not applied in this bond_force kernel. Placeholder argument.
     * regimes - Not applied in this bond_force kernel. Placeholder argument.
     * plus_cs - Not applied in this bond_force kernel. Placeholder argument.
     * local_cache_x - local (local_size) array to store the x components of the bond forces.
     * local_cache_y - local (local_size) array to store the y components of the bond forces.
     * local_cache_z - local (local_size) array to store the z components of the bond forces.
     * bond_stiffness - The bond stiffness.
     * critical_stretch - The critical stretch, at and above which bonds will be broken.
     * fc_scale - scale factor appied to the force bondary conditions.
     * nregimes - Not applied in this bond_force kernel. Placeholder argument. */
    // global_id is the bond number
    const int global_id = get_global_id(0);
    // local_id is the LOCAL node id in range [0, max_neigh] of a node in this parent node's family
	const int local_id = get_local_id(0);
    // local_size is the max_neigh, usually 128 or 256 depending on the problem
    const int local_size = get_local_size(0);
	// group_id is the node i
	const int node_id_i = get_group_id(0);

	// Access local node within node_id_i's horizon with corresponding node_id_j,
	const int node_id_j = nlist[global_id];

	// If bond is not broken
	if (node_id_j != -1) {
		const double xi_x = r0[3 * node_id_j + 0] - r0[3 * node_id_i + 0];
		const double xi_y = r0[3 * node_id_j + 1] - r0[3 * node_id_i + 1];
		const double xi_z = r0[3 * node_id_j + 2] - r0[3 * node_id_i + 2];

		const double xi_eta_x = u[3 * node_id_j + 0] - u[3 * node_id_i + 0] + xi_x;
		const double xi_eta_y = u[3 * node_id_j + 1] - u[3 * node_id_i + 1] + xi_y;
		const double xi_eta_z = u[3 * node_id_j + 2] - u[3 * node_id_i + 2] + xi_z;

		const double xi = sqrt(xi_x * xi_x + xi_y * xi_y + xi_z * xi_z);
		const double y = sqrt(xi_eta_x * xi_eta_x + xi_eta_y * xi_eta_y + xi_eta_z * xi_eta_z);
		const double s = (y -  xi)/ xi;

        // Check for state of bonds here, and break it if necessary
		if (s < critical_stretch) {
            const double cx = xi_eta_x / y;
		    const double cy = xi_eta_y / y;
		    const double cz = xi_eta_z / y;

		    const double f = s * bond_stiffness * vols[node_id_j];
            // Copy bond forces into local memory
		    local_cache_x[local_id] = f * cx;
		    local_cache_y[local_id] = f * cy;
		    local_cache_z[local_id] = f * cz;
		}
        else {
            // bond is broken
			nlist[global_id] = -1;  // Break the bond
            local_cache_x[local_id] = 0.00;
            local_cache_y[local_id] = 0.00;
            local_cache_z[local_id] = 0.00;
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
        double const force_x = local_cache_x[0];
        double const force_y = local_cache_y[0];
        double const force_z = local_cache_z[0];
        // Update body forces in each direction
        body_force[3 * node_id_i + 0] = force_x;
        body_force[3 * node_id_i + 1] = force_y;
        body_force[3 * node_id_i + 2] = force_z;
        // Update forces in each direction
        force[3 * node_id_i + 0] = (fc_types[3 * node_id_i + 0] == 0 ? force_x : (force_x + fc_scale * fc_values[3 * node_id_i + 0]));
        force[3 * node_id_i + 1] = (fc_types[3 * node_id_i + 1] == 0 ? force_y : (force_y + fc_scale * fc_values[3 * node_id_i + 1]));
        force[3 * node_id_i + 2] = (fc_types[3 * node_id_i + 2] == 0 ? force_z : (force_z + fc_scale * fc_values[3 * node_id_i + 2]));
    }
}


__kernel void
	bond_force2(
    __global double const* u,
    __global double* force,
    __global double* body_force,
    __global double const* r0,
    __global double const* vols,
	__global int* nlist,
    __global int const* fc_types,
    __global double const* fc_values,
    __global double const* stiffness_corrections,
    __global int const* bond_types,
    __global int* regimes,
    __global float const* plus_cs,
    __local double* local_cache_x,
    __local double* local_cache_y,
    __local double* local_cache_z,
    double bond_stiffness,
    double critical_stretch,
    double fc_scale,
    int nregimes
	) {
    /* Calculate the force due to bonds on each node.
     *
     * This bond_force function is for the simple case of stiffness corrections and no bond types.
     *
     * u - An (n,3) array of the current displacements of the particles.
     * force - An (n,3) array of the current forces on the particles.
     * body_force - An (n,3) array of the current internal body forces of the particles.
     * r0 - An (n,3) array of the coordinates of the nodes in the initial state.
     * vols - the volumes of each of the nodes.
     * nlist - An (n, local_size) array containing the neighbour lists,
     *     a value of -1 corresponds to a broken bond.
     * fc_types - An (n,3) array of force boundary condition types,
     *     a value of 0 denotes a particle that is not externally loaded.
     * fc_values - An (n,3) array of the force boundary condition values applied to particles.
     * stiffness_corrections - An (n, local_size) array of bond stiffness correction
     *     factors multiplied by the partial volume correction factors.
     * bond_types - Not applied in this bond_force kernel. Placeholder argument.
     * regimes - Not applied in this bond_force kernel. Placeholder argument.
     * plus_cs - Not applied in this bond_force kernel. Placeholder argument.
     * local_cache_x - local (local_size) array to store the x components of the bond forces.
     * local_cache_y - local (local_size) array to store the y components of the bond forces.
     * local_cache_z - local (local_size) array to store the z components of the bond forces.
     * bond_stiffness - The bond stiffness.
     * critical_stretch - The critical stretch, at and above which bonds will be broken.
     * fc_scale - scale factor appied to the force bondary conditions.
     * nregimes - Not applied in this bond_force kernel. Placeholder argument. */
    // global_id is the bond number
    const int global_id = get_global_id(0);
    // local_id is the LOCAL node id in range [0, max_neigh] of a node in this parent node's family
	const int local_id = get_local_id(0);
    // local_size is the max_neigh, usually 128 or 256 depending on the problem
    const int local_size = get_local_size(0);
	// group_id is node i
	const int node_id_i = get_group_id(0);

	// Access local node within node_id_i's horizon with corresponding node_id_j,
	const int node_id_j = nlist[global_id];

	// If bond is not broken
	if (node_id_j != -1) {
		const double xi_x = r0[3 * node_id_j + 0] - r0[3 * node_id_i + 0];
		const double xi_y = r0[3 * node_id_j + 1] - r0[3 * node_id_i + 1];
		const double xi_z = r0[3 * node_id_j + 2] - r0[3 * node_id_i + 2];

		const double xi_eta_x = u[3 * node_id_j + 0] - u[3 * node_id_i + 0] + xi_x;
		const double xi_eta_y = u[3 * node_id_j + 1] - u[3 * node_id_i + 1] + xi_y;
		const double xi_eta_z = u[3 * node_id_j + 2] - u[3 * node_id_i + 2] + xi_z;

		const double xi = sqrt(xi_x * xi_x + xi_y * xi_y + xi_z * xi_z);
		const double y = sqrt(xi_eta_x * xi_eta_x + xi_eta_y * xi_eta_y + xi_eta_z * xi_eta_z);
		const double s = (y -  xi)/ xi;

        // Check for state of bonds here, and break it if necessary
		if (s < critical_stretch) {
            const double cx = xi_eta_x / y;
		    const double cy = xi_eta_y / y;
		    const double cz = xi_eta_z / y;

		    const double f = s * bond_stiffness * stiffness_corrections[global_id] * vols[node_id_j];
            // Copy bond forces into local memory
		    local_cache_x[local_id] = f * cx;
		    local_cache_y[local_id] = f * cy;
		    local_cache_z[local_id] = f * cz;
		}
        else {
            // bond is broken
			nlist[global_id] = -1;  // Break the bond
            local_cache_x[local_id] = 0.00;
            local_cache_y[local_id] = 0.00;
            local_cache_z[local_id] = 0.00;
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
        double const force_x = local_cache_x[0];
        double const force_y = local_cache_y[0];
        double const force_z = local_cache_z[0];
        // Update body forces in each direction
        body_force[3 * node_id_i + 0] = force_x;
        body_force[3 * node_id_i + 1] = force_y;
        body_force[3 * node_id_i + 2] = force_z;
        // Update forces in each direction
        force[3 * node_id_i + 0] = (fc_types[3 * node_id_i + 0] == 0 ? force_x : (force_x + fc_scale * fc_values[3 * node_id_i + 0]));
        force[3 * node_id_i + 1] = (fc_types[3 * node_id_i + 1] == 0 ? force_y : (force_y + fc_scale * fc_values[3 * node_id_i + 1]));
        force[3 * node_id_i + 2] = (fc_types[3 * node_id_i + 2] == 0 ? force_z : (force_z + fc_scale * fc_values[3 * node_id_i + 2]));
    }
}


__kernel void
	bond_force3(
    __global double const* u,
    __global double* force,
    __global double* body_force,
    __global double const* r0,
    __global double const* vols,
	__global int* nlist,
    __global int const* fc_types,
    __global double const* fc_values,
    __global double const* stiffness_corrections,
    __global int const* bond_types,
    __global int* regimes,
    __global double const* plus_cs,
    __local double* local_cache_x,
    __local double* local_cache_y,
    __local double* local_cache_z,
    __global double* bond_stiffness,
    __global double* critical_stretch,
    double fc_scale,
    int nregimes
	) {
    /* Calculate the force due to bonds on each node.
     *
     * This bond_force function is for the case of no stiffness corrections and bond types.
     *
     * u - An (n,3) array of the current displacements of the particles.
     * force - An (n,3) array of the current forces on the particles.
     * body_force - An (n,3) array of the current internal body forces of the particles.
     * r0 - An (n,3) array of the coordinates of the nodes in the initial state.
     * vols - the volumes of each of the nodes.
     * nlist - An (n, local_size) array containing the neighbour lists,
     *     a value of -1 corresponds to a broken bond.
     * fc_types - An (n,3) array of force boundary condition types,
     *     a value of 0 denotes a particle that is not externally loaded.
     * fc_values - An (n,3) array of the force boundary condition values applied to particles.
     * stiffness_corrections - Not applied in this bond_force kernel. Placeholder argument.
     * bond_types - An (n, local_size) array of bond types.
     * regimes - An (n, local_size) array of the bonds' current regime in the damage model.
     * plus_cs - 'c' in 'y=mx+c' of the linear damage model regime.
     * local_cache_x - local (local_size) array to store the x components of the bond forces.
     * local_cache_y - local (local_size) array to store the y components of the bond forces.
     * local_cache_z - local (local_size) array to store the z components of the bond forces.
     * bond_stiffness - The bond stiffness.
     * critical_stretch - The critical stretch, at and above which bonds will be broken.
     * fc_scale - scale factor appied to the force bondary conditions.
     * nregimes - Total number of regimes in the damage model. */
    // global_id is the bond number
    const int global_id = get_global_id(0);
    // local_id is the LOCAL node id in range [0, max_neigh] of a node in this parent node's family
	const int local_id = get_local_id(0);
    // local_size is the max_neigh, usually 128 or 256 depending on the problem
    const int local_size = get_local_size(0);
    // group_id is node i
	const int node_id_i = get_group_id(0);


	// Access local node within node_id_i's horizon with corresponding node_id_j,
	const int node_id_j = nlist[global_id];

    // Find bond type, which chooses the damage model
    const int bond_type = bond_types[global_id];
    int regime = regimes[global_id];
    const double current_critical_stretch = critical_stretch[bond_type * nregimes + regime];

	// If bond is not broken
	if (node_id_j != -1) {
		const double xi_x = r0[3 * node_id_j + 0] - r0[3 * node_id_i + 0];
		const double xi_y = r0[3 * node_id_j + 1] - r0[3 * node_id_i + 1];
		const double xi_z = r0[3 * node_id_j + 2] - r0[3 * node_id_i + 2];

		const double xi_eta_x = u[3 * node_id_j + 0] - u[3 * node_id_i + 0] + xi_x;
		const double xi_eta_y = u[3 * node_id_j + 1] - u[3 * node_id_i + 1] + xi_y;
		const double xi_eta_z = u[3 * node_id_j + 2] - u[3 * node_id_i + 2] + xi_z;

		const double xi = sqrt(xi_x * xi_x + xi_y * xi_y + xi_z * xi_z);
		const double y = sqrt(xi_eta_x * xi_eta_x + xi_eta_y * xi_eta_y + xi_eta_z * xi_eta_z);
		const double s = (y -  xi)/ xi;

        // Check for state of bonds
		if (s < current_critical_stretch) {
            // Check if the bond has entered the previous regime
            if (regime > 0) {
                const double previous_critical_stretch = critical_stretch[bond_type * nregimes + regime - 1];
                if (s < previous_critical_stretch) {
                    // bond enters previous regime
                    regime -= 1;
                    regimes[global_id] = regime;
                }
            }
		}
        else {
            // Bond enters the next regime
            regime += 1;
            regimes[global_id] = regime;
        }
        // Break bond if necessary
        if (regime >= nregimes) {
            nlist[global_id] = -1;  // Break the bond
            local_cache_x[local_id] = 0.00;
            local_cache_y[local_id] = 0.00;
            local_cache_z[local_id] = 0.00;
        }
        else{
            const double cx = xi_eta_x / y;
            const double cy = xi_eta_y / y;
            const double cz = xi_eta_z / y;

            const double f = (s * bond_stiffness[bond_type * nregimes + regime] + plus_cs[bond_type * nregimes + regime]) * vols[node_id_j];
            // Copy bond forces into local memory
            local_cache_x[local_id] = f * cx;
            local_cache_y[local_id] = f * cy;
            local_cache_z[local_id] = f * cz;
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
        double const force_x = local_cache_x[0];
        double const force_y = local_cache_y[0];
        double const force_z = local_cache_z[0];
        // Update body forces in each direction
        body_force[3 * node_id_i + 0] = force_x;
        body_force[3 * node_id_i + 1] = force_y;
        body_force[3 * node_id_i + 2] = force_z;
        // Update forces in each direction
        force[3 * node_id_i + 0] = (fc_types[3 * node_id_i + 0] == 0 ? force_x : (force_x + fc_scale * fc_values[3 * node_id_i + 0]));
        force[3 * node_id_i + 1] = (fc_types[3 * node_id_i + 1] == 0 ? force_y : (force_y + fc_scale * fc_values[3 * node_id_i + 1]));
        force[3 * node_id_i + 2] = (fc_types[3 * node_id_i + 2] == 0 ? force_z : (force_z + fc_scale * fc_values[3 * node_id_i + 2]));
    }
}


__kernel void
	bond_force4(
    __global double const* u,
    __global double* force,
    __global double* body_force,
    __global double const* r0,
    __global double const* vols,
	__global int* nlist,
    __global int const* fc_types,
    __global double const* fc_values,
    __global double const* stiffness_corrections,
    __global int const* bond_types,
    __global int* regimes,
    __global double const* plus_cs,
    __local double* local_cache_x,
    __local double* local_cache_y,
    __local double* local_cache_z,
    __global double* bond_stiffness,
    __global double* critical_stretch,
    double fc_scale,
    int nregimes
	) {
    /* Calculate the force due to bonds on each node.
     *
     * This bond_force function is for the case of stiffness corrections and bond types.
     *
     * u - An (n,3) array of the current displacements of the particles.
     * force - An (n,3) array of the current forces on the particles.
     * body_force - An (n,3) array of the current internal body forces of the particles.
     * r0 - An (n,3) array of the coordinates of the nodes in the initial state.
     * vols - the volumes of each of the nodes.
     * nlist - An (n, local_size) array containing the neighbour lists,
     *     a value of -1 corresponds to a broken bond.
     * fc_types - An (n,3) array of force boundary condition types,
     *     a value of 0 denotes a particle that is not externally loaded.
     * fc_values - An (n,3) array of the force boundary condition values applied to particles.
     * stiffness_corrections - An (n, local_size) array of bond stiffness correction factors.
     * bond_types - An (n, local_size) array of bond types.
     * regimes - An (n, local_size) array of the bonds' current regime in the damage model.
     * plus_cs - 'c' in 'y=mx+c' of the linear damage model regime.
     * local_cache_x - local (local_size) array to store the x components of the bond forces.
     * local_cache_y - local (local_size) array to store the y components of the bond forces.
     * local_cache_z - local (local_size) array to store the z components of the bond forces.
     * bond_stiffness - The bond stiffness.
     * critical_stretch - The critical stretch, at and above which bonds will be broken.
     * fc_scale - scale factor appied to the force bondary conditions.
     * nregimes - Total number of regimes in the damage model. */
    // global_id is the bond number
    const int global_id = get_global_id(0);
    // local_id is the LOCAL node id in range [0, max_neigh] of a node in this parent node's family
	const int local_id = get_local_id(0);
    // local_size is the max_neigh, usually 128 or 256 depending on the problem
    const int local_size = get_local_size(0);
    // group_id is node i
	const int node_id_i = get_group_id(0);

	// Access local node within node_id_i's horizon with corresponding node_id_j,
	const int node_id_j = nlist[global_id];

    // Find bond type, which chooses the damage model
    const int bond_type = bond_types[global_id];
    int regime = regimes[global_id];
    const double current_critical_stretch = critical_stretch[bond_type * nregimes + regime];

	// If bond is not broken
	if (node_id_j != -1) {
		const double xi_x = r0[3 * node_id_j + 0] - r0[3 * node_id_i + 0];
		const double xi_y = r0[3 * node_id_j + 1] - r0[3 * node_id_i + 1];
		const double xi_z = r0[3 * node_id_j + 2] - r0[3 * node_id_i + 2];

		const double xi_eta_x = u[3 * node_id_j + 0] - u[3 * node_id_i + 0] + xi_x;
		const double xi_eta_y = u[3 * node_id_j + 1] - u[3 * node_id_i + 1] + xi_y;
		const double xi_eta_z = u[3 * node_id_j + 2] - u[3 * node_id_i + 2] + xi_z;

		const double xi = sqrt(xi_x * xi_x + xi_y * xi_y + xi_z * xi_z);
		const double y = sqrt(xi_eta_x * xi_eta_x + xi_eta_y * xi_eta_y + xi_eta_z * xi_eta_z);
		const double s = (y -  xi)/ xi;

        // Check for state of bonds
		if (s < current_critical_stretch) {
            // Check if the bond has entered the previous regime
            if (regime > 0) {
                const double previous_critical_stretch = critical_stretch[bond_type * nregimes + regime - 1];
                if (s < previous_critical_stretch) {
                    // bond enters previous regime
                    regime -= 1;
                    regimes[global_id] = regime;
                }
            }
		}
        else {
            // Bond enters the next regime
            regime += 1;
            regimes[global_id] = regime;
        }
        // Break bond if necessary
        if (regime >= nregimes) {
            nlist[global_id] = -1;  // Break the bond
            local_cache_x[local_id] = 0.00;
            local_cache_y[local_id] = 0.00;
            local_cache_z[local_id] = 0.00;
        }
        else{
            const double cx = xi_eta_x / y;
            const double cy = xi_eta_y / y;
            const double cz = xi_eta_z / y;

            const double f = (s * bond_stiffness[bond_type * nregimes + regime] + plus_cs[bond_type * nregimes + regime]) * stiffness_corrections[global_id] * vols[node_id_j];
            // Copy bond forces into local memory
            local_cache_x[local_id] = f * cx;
            local_cache_y[local_id] = f * cy;
            local_cache_z[local_id] = f * cz;
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
        double const force_x = local_cache_x[0];
        double const force_y = local_cache_y[0];
        double const force_z = local_cache_z[0];
        // Update body forces in each direction
        body_force[3 * node_id_i + 0] = force_x;
        body_force[3 * node_id_i + 1] = force_y;
        body_force[3 * node_id_i + 2] = force_z;
        // Update forces in each direction
        force[3 * node_id_i + 0] = (fc_types[3 * node_id_i + 0] == 0 ? force_x : (force_x + fc_scale * fc_values[3 * node_id_i + 0]));
        force[3 * node_id_i + 1] = (fc_types[3 * node_id_i + 1] == 0 ? force_y : (force_y + fc_scale * fc_values[3 * node_id_i + 1]));
        force[3 * node_id_i + 2] = (fc_types[3 * node_id_i + 2] == 0 ? force_z : (force_z + fc_scale * fc_values[3 * node_id_i + 2]));
    }
}


__kernel void damage(
        __global int const *nlist,
		__global int const *family,
        __global int *n_neigh,
        __global double *damage,
        __local double* local_cache
    )
{
    /* Calculate the damage of each node.
     *
     * nlist - An (n, local_size) array containing the neighbour lists,
     *     a value of -1 corresponds to a broken bond.
     * family - An (n) array of the initial number of neighbours for each node.
     * n_neigh - An (n) array of the number of neighbours (particles bound) for
     *     each node.
     * damage - An (n) array of the damage for each node. 
     * local_cache - local (local_size) array to store the bond breakages.*/
    int global_id = get_global_id(0); 
    int local_id = get_local_id(0); 
    int local_size = get_local_size(0); 
    
    //Copy values into local memory 
    local_cache[local_id] = nlist[global_id] != -1 ? 1.00 : 0.00; 

    //Wait for all threads to catch up 
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int i = local_size/2; i > 0; i /= 2){
        if(local_id < i){
            local_cache[local_id] += local_cache[local_id + i];
        } 
        //Wait for all threads to catch up 
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (!local_id) {
        //Get the reduced damages
        int node_id_i = get_group_id(0);
        // Update damage and n_neigh
        int neighbours = local_cache[0];
        n_neigh[node_id_i] = neighbours;
        damage[node_id_i] = 1.00 - (double) neighbours / (double) (family[node_id_i]);
    }
}

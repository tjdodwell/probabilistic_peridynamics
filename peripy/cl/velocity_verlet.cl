#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void
	update_displacement(
        __global double const* force,
        __global double* u,
        __global double* ud,
        __global double* udd,
        __global int const* bc_types,
		__global double const* bc_values,
        __global double const* densities,
        double bc_scale,
        double damping,
        double dt
	){
    /* Calculate the dispalcement and velocity of each node using an
     * Velocity Verlet integrator.
     *
     * force - An (n,3) array of the forces of each node.
     * u - An (n,3) array of the current displacements of each node.
     * ud - An (n,3) array of the current velocities of each node.
     * udd - An (n,3) array of the accelerations of each node.
     * bc_types - An (n,3) array of the boundary condition types.
     * bc_values - An (n,3) array of the boundary condition values applied to the nodes.
     * densties - An (n,3) array of the density values of the nodes.
     * bc_scale - The scalar value applied to the displacement BCs.
     * damping - The dynamic relaxation damping constant in [kg/(m^3 s)].
     * dt - The time step in [s]. */
	const int i = get_global_id(0);

    double const ud1 = ud[i] + (dt / 2) * udd[i]; // Half-step velocity
    double const udd1 = (force[i] - damping * ud1) / densities[i];
    ud[i] = ud1 + (dt / 2) * udd1; // Full-step velocity
    udd[i] = udd1;
    u[i] = (bc_types[i] == 0 ? (u[i] + dt * (ud[i] + (dt / 2) * udd1)) : (bc_scale * bc_values[i]));
}

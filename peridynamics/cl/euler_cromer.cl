#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void
	update_displacement(
        __global double const* force,
        __global double* u,
        __global double* ud,
        __global int const* bc_types,
		__global double const* bc_values,
        __global double const* densities,
        double bc_scale,
        double dt,
        double damping
	){
    /* Calculate the dispalcement and velocity of each node using an
     * Euler Cromer integrator.
     *
     * force - An (n,3) array of the forces of each node.
     * u - An (n,3) array of the current displacements of each node.
     * ud - An (n,3) array of the current velocities of each node.
     * bc_values - An (n,3) array of the boundary condition values applied to the nodes.
     * densties - An (n,3) array of the density values of the nodes.
     * bc_scale - The scalar value applied to the displacement BCs.
     * dt - The time step in [s].
     * damping - The damping constant in [kg/(m^3 s)] */
	const int i = get_global_id(0);

    double udd = (force[i] - damping * ud[i]) / densities[i];
    ud[i] += udd * dt;
    u[i] = (bc_types[i] == 0 ? (u[i] + dt * ud[i]) : (bc_scale * bc_values[i]));
}

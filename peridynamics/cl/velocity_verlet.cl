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
     * Euler Cromer integrator.
     *
     * force - An (n,3) array of the forces of each node.
     * u - An (n,3) array of the current displacements of each node at time t.
     * u1 - An (n,3) array of the displacements of each node at the current time t + dt/2.
     * ud - An (n,3) array of the current velocities of each node at time t.
     * ud1 - An (n,3) array of the velocities of each node at time t + dt/2.
     * udd - An (n,3) array of the current accelerations of each node at time t.
     * bc_values - An (n,3) array of the boundary condition values applied to the nodes.
     * densties - An (n,3) array of the density values of the nodes.
     * bc_scale - The scalar value applied to the displacement BCs.
     * dt - The time step in [s].
     * damping - The damping constant in [kg/(m^3 s)] */
	const int i = get_global_id(0);

    double const udd1 = (force[i] - damping * ud[i]) / densities[i];
    ud[i] += ud[i] + (dt / 2) * (udd1 + udd[i]);
    udd[i] = udd1;
    u[i] = (bc_types[i] == 0 ? (u[i] + dt * (ud[i] + (dt / 2) * udd1)) : (bc_scale * bc_values[i]));
}

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void
	update_displacement(
    	__global double const* force,
    	__global double* u,
		__global int const* bc_types,
		__global double const* bc_values,
		double bc_scale,
        double dt
	){
    /* Calculate the displacement of each node using an Euler
     * integrator.
     *
     * force - An (n,3) array of the forces of each node.
     * u - An (n,3) array of the current displacements of each node.
     * BC_types - An (n,3) array of the boundary condition types...
     * a value of 0 denotes an unconstrained node.
     * bc_values - An (n,3) array of the boundary condition values applied to the nodes.
     * bc_scale - The scalar value applied to the displacement BCs. */
	const int i = get_global_id(0);

	u[i] = (bc_types[i] == 0 ? (u[i] + dt * force[i]) : (bc_scale * bc_values[i]));
}

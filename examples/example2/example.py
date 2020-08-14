"""
A simple, 3D peridynamics simulation example.

This example is a 1.65m x 0.25m x 0.6m plain concrete canteliver beam with no
pre-crack subjected to force controlled loading on the  right-hand side of the
beam which linearly increases up to 45kN.

In this example, the first time the volume, family and connectivity of the
model are calculated, they are also stored in file '1650beam13539_model.h5'.
In subsequent simulations, the arrays are loaded from this h5 file instead of
being calculated again, therefore reducing the overhead of initiating the
model.
"""
import argparse
import cProfile
from io import StringIO
import numpy as np
import pathlib
from peridynamics import Model
from peridynamics.integrators import VelocityVerletCL
from peridynamics.utilities import read_array as read_model
from pstats import SortKey, Stats


# The .msh file is a finite element mesh generated with  a finite
# element mesh generator. '1650beam5153.msh' was generated with gmsh and
# contains 13539 nodes
mesh_file = pathlib.Path(__file__).parent.absolute() / '1650beam13539.msh'


def is_tip(x):
    """
    Return a boolean list of tip types for each cartesian direction.

    Returns a boolean list, whose elements are True when the particle is to
    be measured for some displacement, velocity, force or acceleration
    in that cartesian direction.

    :arg x: Particle coordinate array of size (3,).
    :type x: :class:`numpy.ndarray`
    """
    # Particle does not live on tip
    tip = [None, None, None]
    # Particle does live on tip
    if x[0] > 1.55:
        tip[2] = 1
    return tip


def is_density(x):
    """
    Return the density of the particle.

    :arg x: Particle coordinate array of size (3,).
    :type x: :class:`numpy.ndarray`

    :returns: density in [kg/m^3]
    :rtype: float
    """
    return 2400.0


def is_displacement_boundary(x):
    """
    Return a boolean list of displacement boundarys for each direction.

    Returns a (3,) boolean list, whose elements are:
        None where there is no boundary condition;
        -1 where the boundary is displacement loaded in negative direction;
        1 where the boundary is displacement loaded in positive direction;
        0 where the boundary is clamped;

    :arg x: Particle coordinate array of size (3,).
    :type x: :class:`numpy.ndarray`
    """
    # Particle does not live on a boundary
    bnd = [None, None, None]
    # Particle does live on a boundary
    if x[0] < 0.1:
        # Clamped one horizon distance from the left hand side
        bnd[0] = 0
        bnd[1] = 0
        bnd[2] = 0
    return bnd


def is_force_boundary(x):
    """
    Return a boolean list of force boundarys for each direction.

    Returns a boolean list, whose elements are:
        None where there is no boundary condition;
        -1 where the boundary is displacement loaded in negative direction;
        1 where the boundary is displacement loaded in positive direction;
        0 where the boundary is clamped;

    :arg x: Particle coordinate array of size (3,).
    :type x: :class:`numpy.ndarray`
    """
    # Particle does not live on a boundary
    bnd = [None, None, None]
    if x[0] > 1.55:
        # Force loaded in negative direction on the right hand side
        bnd[2] = -1
    return bnd


def main():
    """Conduct a peridynamics simulation."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', action='store_const', const=True)
    args = parser.parse_args()

    write_path_model = (pathlib.Path(__file__).parent.absolute() / str(
        "1650beam13539_model.h5"))

    # Constants
    # The average one-dimensional grid separation between nodes along an axis
    dx = 0.025
    # Following convention, the horizon distance is taken as just over 3
    # times the grid separation between nodes
    horizon = dx * np.pi
    # Youngs modulus of concrete
    youngs_modulus = 1. * 22e9
    # In bond-based peridynamics, the Poission ratio takes a value of 0.25
    poisson_ratio = 0.25
    # Strain energy release rate of concrete (approximate)
    strain_energy_release_rate = 100.0
    # Critical stretch
    critical_stretch = np.power(
        np.divide(
            5 * strain_energy_release_rate, 6 * youngs_modulus * horizon), (
                1. / 2))
    bulk_modulus = youngs_modulus / (3 * (1 - 2 * poisson_ratio))
    bond_stiffness = (18.00 * bulk_modulus) / (np.pi * np.power(horizon, 4))

    if args.profile:
        profile = cProfile.Profile()
        profile.enable()

    # Increasing the dynamic relaxation damping constant to a critical value
    # will help the system to converge to the equilibrium steady-state solution
    damping = 0.0
    # Stable time step (What happens if you increase or decrease it?)
    dt = 1.5e-5
    integrator = VelocityVerletCL(dt=dt, damping=damping)

    # Try reading connectivity, bond_types and stiffness_correction files from
    # the file ./1650beam13539_model.h5
    # volume is an (nnodes, ) :class:`np.ndarray` of nodal volumes, where
    # nnodes is the number of nodes in the .msh file
    volume = read_model(write_path_model, "volume")
    # density is an (nnodes, ) :class:`np.ndarray` of nodal densities
    density = read_model(write_path_model, "density")
    # family is an (nnodes, ) :class:`np.ndarray` of initial number of
    # neighbours for each node
    family = read_model(write_path_model, "family")
    # nlist is an (nnodes, max_neigh) :class:`np.ndarray` of the neighbours
    # for each node. Each neigbour is given an integer i.d. in the range
    # [0, nnodes). max_neigh is atleast as large as np.max(family)
    nlist = read_model(write_path_model, "nlist")
    # n_neigh is an (nnodes, ) :class:`np.ndarray` of current number of
    # neighbours for each node.
    n_neigh = read_model(write_path_model, "n_neigh")
    # The connectivity of the model is the tuple (nlist, n_neigh)
    if ((nlist is not None) and (n_neigh is not None)):
        connectivity = (nlist, n_neigh)
    else:
        connectivity = None

    if ((volume is not None) and
            (density is not None) and
            (family is not None) and
            (connectivity is not None)):
        # Model has been initiated before, so to avoid calculating volume,
        # family and connectivity arrays again, we can pass them as arguments
        # to the Model class
        model = Model(
            mesh_file, integrator=integrator, horizon=horizon,
            critical_stretch=critical_stretch, bond_stiffness=bond_stiffness,
            dimensions=3, family=family,
            volume=volume, connectivity=connectivity,
            density=density,
            is_displacement_boundary=is_displacement_boundary,
            is_force_boundary=is_force_boundary,
            is_tip=is_tip)
    else:
        # This is the first time that Model has been initiated, so the volume,
        # family and connectivity = (nlist, n_neigh) arrays will be calculated
        # and written to the file at location "write_path_model"
        model = Model(
            mesh_file, integrator=integrator, horizon=horizon,
            critical_stretch=critical_stretch, bond_stiffness=bond_stiffness,
            dimensions=3,
            is_density=is_density,
            is_displacement_boundary=is_displacement_boundary,
            is_force_boundary=is_force_boundary,
            is_tip=is_tip,
            write_path=write_path_model)

    # The force boundary condition magnitudes linearly increment in
    # time with a max force rate of 45kN over 5000 time-steps
    force_bc_array = np.linspace(0, 45000, 5000)

    # Run the simulation
    # Use e.g. paraview to view the output .vtk files of simulate
    (u, damage, connectivity, force, ud, damage_sum_data,
     tip_displacement_data, tip_velocity_data, tip_acceleration_data,
     tip_force_data, tip_body_force_data) = model.simulate(
        steps=5000,
        force_bc_magnitudes=force_bc_array,
        write=200
        )
    # Try plotting tip_body_force_data against tip_displacement_data

    # Note that bond_stiffness and critical_stretch can be changed without
    # re-initialising :class `Model`:, e.g.
    # >>> _* = model.simulate(
    # >>>     bond_stiffness=<my_new_bond_stiffness_value>,
    # >>>     critical_stretch=<my_new_critical_stretch_value>,
    # >>>     steps=5000,
    # >>>     force_bc_magnitudes=force_bc_array,
    # >>>     write=200
    # >>>     )
    # Try experimenting with different values of the peridynamics parameters
    # The horizon distance cannot be changed without re-initialising
    # :class:`Model`, since it controls the connectivity and family of the
    # model

    if args.profile:
        profile.disable()
        s = StringIO()
        stats = Stats(profile, stream=s).sort_stats(SortKey.CUMULATIVE)
        stats.print_stats(.05)
        print(s.getvalue())


if __name__ == "__main__":
    main()

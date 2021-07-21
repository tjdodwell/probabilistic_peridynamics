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
from peripy import Model
from peripy.integrators import VelocityVerletCL
from peripy.utilities import read_array as read_model
from pstats import SortKey, Stats


# The .msh file is a finite element mesh generated with  a finite
# element mesh generator. '1650beam5153.msh' was generated with gmsh and
# contains 13539 particles
mesh_file = pathlib.Path(__file__).parent.absolute() / '1650beam13539.msh'


def is_tip(x):
    """
    Return a boolean list of tip types for each cartesian direction.

    Returns a boolean list, whose elements are not None when the particle
    resides on a 'tip' to be measured for some displacement, velocity,
    acceleration, force or body_force in that cartesian direction. The value
    of the element of the list can be a string or an int, which is a flag for
    the tip type that the particle resides on. If a particle resides on more
    than one tip, then any of the list elements can be a tuple of tip types**.

    :arg x: Particle coordinate array of size (3,).
    :type x: :class:`numpy.ndarray`

    :returns: A (3,) list of tip types.
    :rtype: List of (tuples of) None, int or string.
    """
    # Particle does not live on a tip
    tip = [None, None, None]
    # Particle does live on a tip to be measured, on the right-hand-side
    # (unconstrained end) of the beam
    if x[0] > 1.55:
        # Measurements are made in the z direction
        tip[2] = 'rhs'
    # **e.g. if a particle resides on the top-right-hand-side
    # >>> if x[2] > 0.5 and x[0] > 1.55:
    # >>>     tip[2] = ('rhs', 'top_rhs')
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
    # Average one-dimensional grid separation between particles along an axis
    dx = 0.025
    # Following convention, the horizon distance is taken as just over 3
    # times the grid separation between particles
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
    # will help the system to converge to the quasi-static steady-state.
    # Try 0 damping
    # >>> damping = 0.0
    damping = 2.5e6
    # Stable time step. Try increasing or decreasing it.
    dt = 1.3e-5
    integrator = VelocityVerletCL(dt=dt, damping=damping)

    # Try reading volume, density, family and connectivity arrays from
    # the file ./1650beam13539_model.h5
    # volume is an (nnodes, ) :class:`np.ndarray` of particle volumes, where
    # nnodes is the number of particles in the .msh file
    volume = read_model(write_path_model, "volume")
    # density is an (nnodes, ) :class:`np.ndarray` of particle densities
    density = read_model(write_path_model, "density")
    # family is an (nnodes, ) :class:`np.ndarray` of initial number of
    # neighbours for each particle
    family = read_model(write_path_model, "family")
    # nlist is an (nnodes, max_neigh) :class:`np.ndarray` of the neighbours
    # for each particle. Each neigbour is given an integer i.d. in the range
    # [0, nnodes). max_neigh is atleast as large as np.max(family)
    nlist = read_model(write_path_model, "nlist")
    # n_neigh is an (nnodes, ) :class:`np.ndarray` of current number of
    # neighbours for each particle.
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
    (u, damage, connectivity, force, ud, data) = model.simulate(
        steps=5000,
        force_bc_magnitudes=force_bc_array,
        write=200
        )

    # Try plotting data['rhs']['body_force'] vs data['rhs']['displacement']
    # Try plotting data['model']['damage'] vs data['model']['step']

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
    # :class:`Model`, since it controls the connectivity of the model

    if args.profile:
        profile.disable()
        s = StringIO()
        stats = Stats(profile, stream=s).sort_stats(SortKey.CUMULATIVE)
        stats.print_stats(.05)
        print(s.getvalue())


if __name__ == "__main__":
    main()

"""
A simple, 2D peridynamics simulation example.

This example is a 1.0m x 1.0m 2D plate with a central pre-crack subjected to
uniform velocity displacements on the left-hand side and right-hand side of
2.5x10^-6 metres per time-step.
"""
import argparse
import cProfile
from io import StringIO
import numpy as np
import pathlib
from peridynamics import Model
from peridynamics.model import initial_crack_helper
from peridynamics.integrators import EulerCL, Euler
from pstats import SortKey, Stats


# The .msh file is a finite element mesh generated with a finite
# element mesh generator. 'test.vtk' was generated with gmsh and
# contains 2113 nodes.
mesh_file = pathlib.Path(__file__).parent.absolute() / "test.vtk"


@initial_crack_helper
def is_crack(x, y):
    """Determine whether a pair of particles define the crack."""
    output = 0
    crack_length = 0.3
    p1 = x
    p2 = y
    if x[0] > y[0]:
        p2 = x
        p1 = y
    # 1e-6 makes it fall one side of central line of particles
    if p1[0] < 0.5 + 1e-6 and p2[0] > 0.5 + 1e-6:
        # draw a straight line between them
        m = (p2[1] - p1[1]) / (p2[0] - p1[0])
        c = p1[1] - m * p1[0]
        # height a x = 0.5
        height = m * 0.5 + c
        if (height > 0.5 * (1 - crack_length)
                and height < 0.5 * (1 + crack_length)):
            output = 1
    return output


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
    # Particle does live on boundary
    if x[0] < 0.15:
        bnd[0] = -1
    elif x[0] > 0.85:
        bnd[0] = 1
    return bnd


def main():
    """Conduct a peridynamics simulation."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', action='store_const', const=True)
    # The --opencl argument toggles between OpenCL and cython implementations
    parser.add_argument('--opencl', action='store_const', const=True)
    args = parser.parse_args()
    if args.profile:
        profile = cProfile.Profile()
        profile.enable()

    if args.opencl:
        # The :class:`peridynamics.integrators.EulerCL` class is the OpenCL
        # implementation of the explicit Euler integration scheme.
        integrator = EulerCL(dt=1e-3)
    else:
        # The :class:`peridynamics.integrators.Euler` class is the cython
        # implementation of the explicit Euler integration scheme.
        integrator = Euler(dt=1e-3)

    # The bond_stiffness, also known as the micromodulus, of the peridynamic
    # bond, using Silling's (2005) derivation for the prototype microelastic
    # brittle (PMB) material model.
    # An arbritrary value of the critical_stretch = 0.005m is used.
    horizon = 0.1
    bond_stiffness = 18.00 * 0.05 / (np.pi * horizon**4)
    # The :class:`peridynamics.model.Model` defines and calculates the
    # connectivity of the model, as well as the boundary conditions and crack.
    model = Model(
        mesh_file, integrator=integrator, horizon=horizon,
        critical_stretch=0.005, bond_stiffness=bond_stiffness,
        is_displacement_boundary=is_displacement_boundary,
        dimensions=2, initial_crack=is_crack)

    # The simulation will have 1000 time steps, and last
    # dt * steps = 1e-3 * 1000 = 1.0 seconds
    steps = 1000

    # The boundary condition magnitudes will be applied at a rate of
    # 2.5e-6 m per time-step, giving a total final displacement (the sum of the
    # left and right hand side) of 5mm.
    displacement_bc_array = np.linspace(2.5e-6, 2.5e-3, steps)

    # The :meth:`Model.simulate` method can be used to conduct a peridynamics
    # simulation. Here it is possible to define the boundary condition
    # magnitude throughout the simulation.
    u, damage, *_ = model.simulate(
        steps=steps,
        displacement_bc_magnitudes=displacement_bc_array,
        write=100)

    if args.profile:
        profile.disable()
        s = StringIO()
        stats = Stats(profile, stream=s).sort_stats(SortKey.CUMULATIVE)
        stats.print_stats(.05)
        print(s.getvalue())


if __name__ == "__main__":
    main()

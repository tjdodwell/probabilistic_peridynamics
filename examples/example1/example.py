"""
A simple, 2D peridynamics simulation example.

This example is a 1.0m x 1.0m 2D plate with a central pre-crack subjected to
uniform velocity displacements on the left-hand side and righ-hand side of
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
from peridynamics.utilities import calc_boundary_conditions_magnitudes
from pstats import SortKey, Stats


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


def is_tip(x):
    """Return if the particle coordinate is a `tip`."""
    # Particle does not live on tip
    tip = [None, None, None]
    # Particle does live on tip
    if x[0] > 0.9:
        tip[0] = 1
    return tip


def is_displacement_boundary(x):
    """
    Return if the particle coordinate is a displacement boundary.

    Function which marks displacement boundary constrained particles. Returns
    a list (3) for constraint in each direction. None is no boundary condition,
    -1 is displacement loaded IN -ve direction, 1 is displacement loaded IN +ve
    direction, 0 is clamped boundary.
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
    parser.add_argument('--opencl', action='store_const', const=True)
    args = parser.parse_args()
    if args.profile:
        profile = cProfile.Profile()
        profile.enable()

    if args.opencl:
        integrator = EulerCL(dt=1e-3)
    else:
        integrator = Euler(dt=1e-3)

    model = Model(
        mesh_file, integrator=integrator, horizon=0.1, critical_stretch=0.005,
        bond_stiffness=18.00 * 0.05 / (np.pi * 0.1**4),
        is_displacement_boundary=is_displacement_boundary,
        is_tip=is_tip, dimensions=2, initial_crack=is_crack)

    # Example function for calculating the boundary conditions magnitudes
    displacement_bc_array, *_ = calc_boundary_conditions_magnitudes(
        steps=1000, max_displacement_rate=0.0000025)

    u, damage, *_ = model.simulate(
        steps=1000,
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

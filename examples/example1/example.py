"""A simple, 2D peridynamics simulation example."""
import argparse
import cProfile
from io import StringIO
import numpy as np
import pathlib
from peridynamics import Model, ModelCL, ModelCLBen
from peridynamics.model import initial_crack_helper
from peridynamics.integrators import Euler
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

def bond_type(x, y):
    """ 
    Determines bond type given pair of node coordinates.
    Usage:
        'plain = 1' will return a plain concrete bond for all bonds, an so a
    plain concrete beam.
        'plain = 0' will return a concrete beam with some rebar as specified
        in "is_rebar()"
    """
    output = 'concrete' # default to concrete
    return output

def boundary_function(model, u, step):
    """
    Apply a load to the system.

    Particles on each of the sides of the system are pulled apart with
    increasing time step.
    """
    load_rate = 0.00001

    u[model.lhs, 1:3] = 0.
    u[model.rhs, 1:3] = 0.

    extension = 0.5 * step * load_rate
    u[model.lhs, 0] = -extension
    u[model.rhs, 0] = extension

    return u


def main():
    """Conduct a peridynamics simulation."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', action='store_const', const=True)
    parser.add_argument('--opencl', action='store_const', const=True)
    parser.add_argument('--ben', action='store_const', const=True)
    args = parser.parse_args()

    if args.profile:
        profile = cProfile.Profile()
        profile.enable()

    if args.opencl:
        if args.ben:
            model = ModelCLBen(mesh_file, horizon=0.1, critical_strain=0.005,
                               elastic_modulus=0.05, initial_crack=is_crack,
                               density=1.0, damping = 1.0,
                               bond_stiffness_concrete = (
                                   np.double((18.00 * 0.05) /
                                                 (np.pi * np.power(0.1, 4)))),
                               critical_strain_concrete = 0.005,
                               crack_length = 0.3,
                               volume_total=1.0,
                               bond_type=bond_type,
                               network_file_name = 'Network_2.vtk',
                               dimensions=2,
                               transfinite=0,
                               precise_stiffness_correction=1,
                               displacement_rate = 0.00001) 
        else:
            model = ModelCL(mesh_file, horizon=0.1, critical_strain=0.005,
                            elastic_modulus=0.05, initial_crack=is_crack)
    else:
        model = Model(mesh_file, horizon=0.1, critical_strain=0.005,
                      elastic_modulus=0.05, initial_crack=is_crack)

    # Set left-hand side and right-hand side of boundary
    model.lhs = np.nonzero(model.coords[:, 0] < 1.5*model.horizon)
    model.rhs = np.nonzero(model.coords[:, 0] > 1.0 - 1.5*model.horizon)

    integrator = Euler(dt=1e-3)

    u, damage, *_ = model.simulate(
        steps=1000,
        integrator=integrator,
        boundary_function=boundary_function,
        write=10000
        )

    if args.profile:
        profile.disable()
        s = StringIO()
        stats = Stats(profile, stream=s).sort_stats(SortKey.CUMULATIVE)
        stats.print_stats(.05)
        print(s.getvalue())


if __name__ == "__main__":
    main()

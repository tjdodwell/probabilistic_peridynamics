"""A simple, 2D peridynamics simulation example."""
import argparse
import cProfile
from io import StringIO
import numpy as np
import pathlib
from peridynamics import Model, ModelCL, ModelCLBen
from peridynamics.model import initial_crack_helper
from peridynamics.integrators import Euler
from peridynamics.utilities import read_array as read_network
from pstats import SortKey, Stats
import shutil
import os

mesh_files = {
    '1650beam792t.msh': '1650beam792t',
    '1650beam2652t.msh': '1650beam2652t',
    '1650beam3570t.msh': '1650beam3570t',
    '1650beam4095t.msh': '1650beam4095t',
    '1650beam6256t.msh': '1650beam6256t',
    '1650beam15840t.msh': '1650beam15840t',
    '1650beam32370t.msh': '1650beam32370t',
    '1650beam74800t.msh': '1650beam74800t',
    '1650beam144900t.msh': '1650beam144900t',
    '1650beam247500t.msh': '1650beam247500t'}

dxs = {
    '1650beam792t.msh': 0.075,
    '1650beam2652t.msh': 0.0485,
    '1650beam3570t.msh': 0.0485,
    '1650beam4095t.msh': 0.0423,
    '1650beam6256t.msh': 0.0359,
    '1650beam15840t.msh': 0.025,
    '1650beam32370t.msh': 0.020,
    '1650beam74800t.msh': 0.015,
    '1650beam144900t.msh': 0.012,
    '1650beam247500t.msh': 0.010}

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
    tip = [2, 2, 2]
    # Particle does live on tip
    if x[0] > 1.65 - 0.01:
        tip[0] = 1
    return tip


def is_boundary(x):
    """
    Return if the particle coordinate is a displacement boundary.

    Function which marks displacement boundary constrained particles
    2 is no boundary condition (the number here is an arbitrary choice)
    -1 is displacement loaded IN -ve direction
    1 is displacement loaded IN +ve direction
    0 is clamped boundary
    """
    # Particle does not live on a boundary
    bnd = [2, 2, 2]
    # Particle does live on a boundary
    if x[0] < 0.01:
        bnd[0] = 0
        bnd[1] = 0
        bnd[2] = 0
    if x[0] > 1.65 - 0.01:
        bnd[2] = -1
    return bnd


def is_forces_boundary(x):
    """
    Return if the particle coordinate is a force boundary.

    Marks types of body force on the particles
    2 is no boundary condition (the number here is an arbitrary choice)
    -1 is force loaded IN -ve direction
    1 is force loaded IN +ve direction
    """
    # Particle does not live on forces boundary
    bnd = [2, 2, 2]
    return bnd


def boundary_function(model, u, step):
    """
    Apply a load to the system.

    Particles on the right hand boundary are pulled downwards with increasing
    timestep.
    """
    load_rate = 5e-9

    u[model.lhs, 1:3] = 0.

    deflection = step * load_rate
    u[model.rhs, 2] = -1. * deflection

    return u


def main():
    """Conduct a peridynamics simulation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("mesh_file_name", help="run example on a given mesh file name")
    parser.add_argument('--profile', action='store_const', const=True)
    parser.add_argument('--opencl', action='store_const', const=True)
    parser.add_argument('--ben', action='store_const', const=True)
    args = parser.parse_args()

    mesh_file = pathlib.Path(__file__).parent.absolute() / args.mesh_file_name
    write_path_solutions = pathlib.Path(__file__).parent.absolute() / "output/"
    write_path_network = (pathlib.Path(__file__).parent.absolute()  / 
                        str(mesh_files[args.mesh_file_name] + "/network.h5"))
    dx = dxs[args.mesh_file_name]

    # Constants
    density = 2400
    youngs_modulus = 1. * 22e9
    poisson_ratio = 0.25
    strain_energy_release_rate = 100
    horizon = dx * np.pi 
    critical_stretch = np.double(np.power(
            np.divide(
                5*strain_energy_release_rate, 6*youngs_modulus*horizon),
            (1./2)
            ))
    bulk_modulus = youngs_modulus/ (3* (1 - 2*poisson_ratio))
    bond_stiffness = (
    np.double((18.00 * bulk_modulus) /
    (np.pi * np.power(horizon, 4)))
    )
    dt = 2.5e-13

    # Try reading connectivity, material_types and stiffness_correction files
    volume = read_network(write_path_network, "volume")
    family = read_network(write_path_network, "family")
    nlist = read_network(write_path_network, "nlist")
    material_types = read_network(write_path_network, "material_types")
    n_neigh = read_network(write_path_network, "n_neigh")
    stiffness_corrections = read_network(
      write_path_network, "stiffness_corrections")

    if ((nlist is not None) and (n_neigh is not None)):
        connectivity = (nlist, n_neigh)
    else:
        connectivity = None

    if args.profile:
          profile = cProfile.Profile()
          profile.enable()

    if args.opencl:
        if args.ben:
            model = ModelCLBen(
                mesh_file, horizon=horizon, critical_stretch=critical_stretch,
                bond_stiffness=bond_stiffness,
                dimensions=3, density=density, dt=dt,
                write_path=write_path_network, family=family, volume=volume,
                connectivity=connectivity, material_types=material_types,
                stiffness_corrections=stiffness_corrections)
        else:
            model = ModelCL(mesh_file, horizon=horizon,
                            critical_stretch=critical_stretch,
                            bond_stiffness=bond_stiffness,
                            dimensions=3, family=family, volume=volume,
                            connectivity=connectivity)
    else:
        model = Model(mesh_file, horizon=horizon,
                      critical_stretch=critical_stretch,
                      bond_stiffness=bond_stiffness,
                      dimensions=3, family=family, volume=volume,
                      connectivity=connectivity)

    # Set left-hand side and right-hand side of boundary
    model.lhs = np.nonzero(model.coords[:, 0] < 0.01)
    model.rhs = np.nonzero(model.coords[:, 0] > 1.65 - 0.01)

    # Delete output directory
    shutil.rmtree(write_path_solutions, ignore_errors=False)
    os.mkdir(write_path_solutions)
    if (args.opencl and args.ben):
        u, damage, *_ = model.simulate(
            steps=1000, is_boundary=is_boundary,
            is_forces_boundary=is_forces_boundary, is_tip=is_tip,
            displacement_rate=5e-9, write=10000,
            write_path=write_path_solutions)
    else:

        integrator = Euler(dt=dt)

        u, damage, *_ = model.simulate(
            steps=1000,
            integrator=integrator,
            boundary_function=boundary_function,
            write=10000,
            write_path=write_path_solutions
            )

    if args.profile:
        profile.disable()
        s = StringIO()
        stats = Stats(profile, stream=s).sort_stats(SortKey.CUMULATIVE)
        stats.print_stats(.05)
        print(s.getvalue())


if __name__ == "__main__":
    main()

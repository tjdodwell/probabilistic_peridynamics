"""A simple, 2D peridynamics simulation example."""
import argparse
import cProfile
from io import StringIO
import numpy as np
import pathlib
from peridynamics import Model
from peridynamics.model import initial_crack_helper
from peridynamics.integrators import Euler, EulerOpenCL
from peridynamics.utilities import read_array as read_model
from peridynamics.utilities import calc_boundary_conditions_magnitudes
from pstats import SortKey, Stats

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
    tip = [None, None, None]
    # Particle does live on tip
    if x[0] > 1.65 - 0.01:
        tip[0] = 1
    return tip


def is_displacement_boundary(x):
    """
    Return if the particle coordinate is a displacement boundary.

    Function which marks displacement boundary constrained particles
    None is no boundary condition
    -1 is displacement loaded in negative direction
    1 is displacement loaded in positive direction
    0 is clamped boundary
    """
    # Particle does not live on a boundary
    bnd = [None, None, None]
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
    None is no boundary condition
    -1 is force loaded in negative direction
    1 is force loaded in positive direction
    """
    # Particle does not live on forces boundary
    bnd = [None, None, None]
    return bnd


def main():
    """Conduct a peridynamics simulation."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mesh_file_name", help="run example on a given mesh file name")
    parser.add_argument('--opencl', action='store_const', const=True)
    parser.add_argument('--profile', action='store_const', const=True)
    args = parser.parse_args()

    mesh_file = pathlib.Path(__file__).parent.absolute() / args.mesh_file_name
    write_path_solutions = pathlib.Path(__file__).parent.absolute()
    write_path_model = (pathlib.Path(__file__).parent.absolute() / str(
        mesh_files[args.mesh_file_name] + "_model.h5"))
    dx = dxs[args.mesh_file_name]

    # Constants
    youngs_modulus = 1. * 22e9
    poisson_ratio = 0.25
    strain_energy_release_rate = 100
    horizon = dx * np.pi
    critical_stretch = np.power(
        np.divide(
            5 * strain_energy_release_rate, 6 * youngs_modulus * horizon), (
                1. / 2))
    bulk_modulus = youngs_modulus / (3 * (1 - 2 * poisson_ratio))
    bond_stiffness = (18.00 * bulk_modulus) / (np.pi * np.power(horizon, 4))

    # Try reading connectivity, material_types and stiffness_correction files
    volume = read_model(write_path_model, "volume")
    family = read_model(write_path_model, "family")
    nlist = read_model(write_path_model, "nlist")
    material_types = read_model(write_path_model, "material_types")
    n_neigh = read_model(write_path_model, "n_neigh")
    stiffness_corrections = read_model(
      write_path_model, "stiffness_corrections")

    if ((nlist is not None) and (n_neigh is not None)):
        connectivity = (nlist, n_neigh)
    else:
        connectivity = None

    if args.profile:
        profile = cProfile.Profile()
        profile.enable()

    if args.opencl:
        integrator = EulerOpenCL(dt=2.5e-13)
    else:
        integrator = Euler(dt=2.5e-13)

    model = Model(
        mesh_file, integrator=integrator, horizon=horizon,
        critical_stretch=critical_stretch, bond_stiffness=bond_stiffness,
        dimensions=3, family=family,
        volume=volume, connectivity=connectivity,
        material_types=material_types,
        stiffness_corrections=stiffness_corrections,
        is_displacement_boundary=is_displacement_boundary,
        is_forces_boundary=is_forces_boundary,
        is_tip=is_tip)

    # Example function for calculating the boundary conditions magnitudes
    displacement_bc_array, *_ = calc_boundary_conditions_magnitudes(
        steps=10000, max_displacement_rate=5e-8)

    u, damage, *_ = model.simulate(
        steps=10000,
        displacement_bc_magnitudes=displacement_bc_array,
        write=1000,
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

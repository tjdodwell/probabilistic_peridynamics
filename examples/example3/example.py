"""A simple, 3D peridynamics simulation example for varying problem sizes."""
import argparse
import cProfile
from io import StringIO
import numpy as np
import pathlib
from peridynamics import Model
from peridynamics.integrators import EulerCromerCL
from peridynamics.utilities import read_array as read_model
from peridynamics.utilities import calc_boundary_conditions_magnitudes
from pstats import SortKey, Stats

mesh_files = {
    '1650beam74800.msh': '1650beam74800',
    '1650beam74800transfinite.msh': '1650beam74800transfinite',
    '1650beam144900.msh': '1650beam144900',
    '1650beam144900transfinite.msh': '1650beam144900transfinite',
    '1650beam247500.msh': '1650beam247500',
    '1650beam247500transfinite.msh': '1650beam247500transfinite'}

dxs = {
    '1650beam74800.msh': 0.015,
    '1650beam74800transfinite.msh': 0.015,
    '1650beam144900.msh': 0.012,
    '1650beam144900transfinite.msh': 0.012,
    '1650beam247500.msh': 0.010,
    '1650beam247500transfinite.msh': 0.010}

build_displacements = {
    '1650beam74800transfinite.msh': 0.0009464,
    '1650beam144900transfinite.msh': 0.0005,
    '1650beam247500transfinite.msh': 0.0005,
    '1650beam74800.msh': 0.0009464,
    '1650beam144900.msh': 0.0005,
    '1650beam247500.msh': 0.0005}

time_steps = {
    '1650beam74800transfinite.msh': 200000,
    '1650beam144900transfinite.msh': 100000,
    '1650beam247500transfinite.msh': 100000,
    '1650beam74800.msh': 200000,
    '1650beam144900.msh': 100000,
    '1650beam247500.msh': 100000}

safety_factors = {
    '1650beam74800transfinite.msh': 2.0,
    '1650beam144900transfinite.msh': 2.0,
    '1650beam247500transfinite.msh': 2.0,
    '1650beam74800.msh': 2.0,
    '1650beam144900.msh': 2.0,
    '1650beam247500.msh': 2.0}


def is_material(x):
    """Determine whether the node is concrete or steel."""
    # y and z coordinates
    x = x[1:]
    bar_centres = [
        # Compressive bars 25mm of cover
        np.array((0.031, 0.031)),
        np.array((0.219, 0.031)),

        # Tensile bars 25mm of cover
        np.array((0.03825, 0.569)),
        np.array((0.21175, 0.569))]

    radius_compression_bar = 0.006
    radius_tension_bar = 0.01325

    radii = [
        radius_compression_bar,
        radius_compression_bar,
        radius_tension_bar,
        radius_tension_bar]

    costs = np.array([np.sum(
        np.square(centre - x) - np.square(radius)) for centre, radius in zip(
                        bar_centres, radii)])
    if np.any(costs <= 0):
        # Node is steel
        return 1
    else:
        # Node is concrete
        return 0


def is_density(x):
    """Determine the density of the node."""
    if is_material(x) == 1:
        density_steel = 7850.0
        return density_steel
    elif is_material(x) == 0:
        density_concrete = 2400.0
        return density_concrete


def is_bond_type(x, y):
    """Determine the type of the bond which defines a damage-model."""
    if is_material(x) and is_material(y):
        # Steel
        return 2
    elif (is_material(x) or is_material(y)):
        # Interface
        return 1
    else:
        # Concrete
        return 0


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


def main():
    """Conduct a peridynamics simulation."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mesh_file_name", help="run example on a given mesh file name")
    parser.add_argument('--profile', action='store_const', const=True)
    args = parser.parse_args()

    mesh_file = pathlib.Path(__file__).parent.absolute() / args.mesh_file_name
    write_path_solutions = pathlib.Path(__file__).parent.absolute()
    write_path_model = (pathlib.Path(__file__).parent.absolute() / str(
        mesh_files[args.mesh_file_name] + "_model.h5"))
    dx = dxs[args.mesh_file_name]

    # Constants
    youngs_modulus_concrete = 1. * 22e9
    youngs_modulus_steel = 1.*210e9
    poisson_ratio = 0.25
    strain_energy_release_rate_concrete = 100
    horizon = dx * np.pi
    critical_stretch_concrete = np.power(
        np.divide(5 * strain_energy_release_rate_concrete,
                  6 * youngs_modulus_concrete * horizon), (1. / 2))
    # Some very large value (assume steel doesn't yield)
    critical_stretch_steel = 10.0
    bulk_modulus_concrete = (youngs_modulus_concrete
                             / (3 * (1 - 2 * poisson_ratio)))
    bond_stiffness_concrete = (18.00 * bulk_modulus_concrete
                               / (np.pi * np.power(horizon, 4)))
    bulk_modulus_steel = (youngs_modulus_steel
                          / (3 * (1 - 2 * poisson_ratio)))
    bond_stiffness_steel = (18.00 * bulk_modulus_steel
                            / np.pi * np.power(horizon, 4))
    bond_stiffness = [[bond_stiffness_concrete],
                      [3.0 * bond_stiffness_concrete],
                      [bond_stiffness_steel]]
    critical_stretch = [[critical_stretch_concrete],
                        [3.0 * critical_stretch_concrete],
                        [critical_stretch_steel]]
    # Try reading connectivity, bond_types and stiffness_correction files
    volume = read_model(write_path_model, "volume")
    density = read_model(write_path_model, "density")
    family = read_model(write_path_model, "family")
    nlist = read_model(write_path_model, "nlist")
    bond_types = read_model(write_path_model, "bond_types")
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

    # Integrator object
    saf_fac = safety_factors[args.mesh_file_name]
    dt = (0.8 * np.power(2.0 * 2400.0 * dx / (
            np.pi
            * np.power(horizon, 2.0)
            * dx
            * bond_stiffness_concrete), 0.5) / saf_fac)
    integrator = EulerCromerCL(dt=dt, damping=2.0e6)

    # Model
    if str('transfinite') in args.mesh_file_name:
        model = Model(
            mesh_file,
            integrator=integrator,
            horizon=horizon,
            critical_stretch=critical_stretch,
            bond_stiffness=bond_stiffness,
            transfinite=1,
            volume_total=1.65 * 0.6 * 0.25,
            write_path=write_path_model,
            connectivity=connectivity,
            family=family,
            volume=volume,
            dimensions=3,
            is_density=is_density,
            is_bond_type=is_bond_type,
            is_displacement_boundary=is_displacement_boundary,
            is_tip=is_tip,
            density=density,
            bond_types=bond_types,
            stiffness_corrections=stiffness_corrections,
            precise_stiffness_correction=0)
    else:
        model = Model(
            mesh_file,
            integrator=integrator,
            horizon=horizon,
            critical_stretch=critical_stretch,
            bond_stiffness=bond_stiffness,
            write_path=write_path_model,
            connectivity=connectivity,
            family=family,
            volume=volume,
            dimensions=3,
            is_density=is_density,
            is_bond_type=is_bond_type,
            is_displacement_boundary=is_displacement_boundary,
            is_tip=is_tip,
            density=density,
            bond_types=bond_types,
            stiffness_corrections=stiffness_corrections,
            precise_stiffness_correction=1)

    # Example function for calculating the boundary conditions magnitudes
    displacement_bc_array, *_ = calc_boundary_conditions_magnitudes(
        steps=time_steps[args.mesh_file_name], max_displacement_rate=1e-8,
        build_displacement=build_displacements[args.mesh_file_name])

    # Simulation
    (u,
     damage,
     connectivity,
     force,
     ud,
     damage_sum_data,
     tip_displacement_data,
     *_) = model.simulate(
        steps=time_steps[args.mesh_file_name],
        displacement_bc_magnitudes=displacement_bc_array,
        write=1000,
        write_path=write_path_solutions
        )
    print('tip_displacement_data', tip_displacement_data)

    if args.profile:
        profile.disable()
        s = StringIO()
        stats = Stats(profile, stream=s).sort_stats(SortKey.CUMULATIVE)
        stats.print_stats(.05)
        print(s.getvalue())


if __name__ == "__main__":
    main()

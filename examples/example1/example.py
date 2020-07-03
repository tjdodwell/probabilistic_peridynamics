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

def is_tip(horizon, x):
    output = 0
    if x[0] > 1.0 - 1. * horizon:
        output = 1
    return output

def is_boundary(horizon, x):
    """
    Function which marks displacement boundary constrained particles
    2 is no boundary condition (the number here is an arbitrary choice)
    -1 is displacement loaded IN -ve direction
    1 is displacement loaded IN +ve direction
    0 is clamped boundary
    """
    # Does not live on a boundary
    bnd = 2
    # Does live on boundary
    if x[0] < 1.5 * horizon:
        bnd = -1
    elif x[0] > 1.0 - 1.5 * horizon:
        bnd = 1
    return bnd

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

def is_forces_boundary(horizon, x):
    """
    Marks types of body force on the particles
    2 is no boundary condition (the number here is an arbitrary choice)
    -1 is force loaded IN -ve direction
    1 is force loaded IN +ve direction
    """
    bnd = [2, 2, 2]
    return bnd

def boundary_function_cl(model, displacement_rate):
    """ 
    Initiates displacement boundary conditions,
    also define the 'tip' (for plotting displacements)
    """
    #initiate containers
    model.bc_types = np.zeros((model.nnodes, model.degrees_freedom), dtype=np.intc)
    model.bc_values = np.zeros((model.nnodes, model.degrees_freedom), dtype=np.float64)
    model.tip_types = np.zeros(model.nnodes, dtype=np.intc)

    # Find the boundary nodes and apply the displacement values
    for i in range(0, model.nnodes):
        # Define boundary types and values
        bnd = is_boundary(model.horizon, model.coords[i][:])
        model.bc_types[i, 0] = np.intc((bnd))
        model.bc_types[i, 1] = np.intc((bnd))
        model.bc_types[i, 2] = np.intc((bnd))
        model.bc_values[i, 0] = np.float64(bnd * 0.5 * displacement_rate)

        # Define tip here
        tip = is_tip(model.horizon, model.coords[i][:])
        model.tip_types[i] = np.intc(tip)

def boundary_forces_function(model):
    """ 
    Initiates boundary forces
    """
    model.force_bc_types = np.zeros((model.nnodes, model.degrees_freedom), dtype=np.intc)
    model.force_bc_values = np.zeros((model.nnodes, model.degrees_freedom), dtype=np.float64)

    # Find the force boundary nodes and find amount of boundary nodes
    num_force_bc_nodes = 0
    for i in range(0, model.nnodes):
        bnd = is_forces_boundary(model.horizon, model.coords[i][:])
        if -1 in bnd:
            num_force_bc_nodes += 1
        elif 1 in bnd:
            num_force_bc_nodes += 1
        model.force_bc_types[i, 0] = np.intc((bnd[0]))
        model.force_bc_types[i, 1] = np.intc((bnd[1]))
        model.force_bc_types[i, 2] = np.intc((bnd[2]))

    model.num_force_bc_nodes = num_force_bc_nodes


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
                               precise_stiffness_correction=2,
                               displacement_rate = 0.00001,
                               dt = 1e-3) 
            boundary_function_cl(model, displacement_rate=0.00001)
            boundary_forces_function(model)
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
        write=100
        )

    if args.profile:
        profile.disable()
        s = StringIO()
        stats = Stats(profile, stream=s).sort_stats(SortKey.CUMULATIVE)
        stats.print_stats(.05)
        print(s.getvalue())

if __name__ == "__main__":
    main()

"""Shared definitions for test modules."""
from ..cl import get_context
from ..model import Model, initial_crack_helper
from ..integrators import Euler, EulerCL
import numpy as np
import pathlib
import pytest


@pytest.fixture(scope="session")
def data_path():
    """Path to the test data directory."""
    path = pathlib.Path(__file__).parent.absolute() / "data"
    return path


context_available = pytest.mark.skipif(
    get_context() is None,
    reason="Suitable OpenCL context required."
    )


@pytest.fixture(scope="session")
def simple_displacement_boundary():
    """Return a simple is_boundary function."""
    def is_boundary(x):
        """Return a simple displacement boundary function."""
        # Particle does not live on a boundary
        bnd = [None, None, None]
        # Particle does live on boundary
        if x[0] < 1.5 * 0.1:
            bnd[0] = -1
            bnd[1] = 0
            bnd[2] = 0
        elif x[0] > 1.0 - 1.5 * 0.1:
            bnd[0] = 1
            bnd[1] = 0
            bnd[2] = 0
        return bnd
    return is_boundary


@initial_crack_helper
def is_crack(x, y):
    """Determine whether a pair of particles define the crack."""
    crack_length = 0.3
    output = 0
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


@pytest.fixture(
    scope="session"
    )
def cython_model(data_path, request, simple_displacement_boundary):
    """Create a simple peridynamics Model object."""
    path = data_path
    mesh_file = path / "example_mesh.vtk"

    # Initiate integrator
    euler = Euler(dt=1e-3)

    # Create model
    model = Model(mesh_file, integrator=euler, horizon=0.1,
                  critical_stretch=0.005,
                  bond_stiffness=18.0 * 0.05 / (np.pi * 0.1**4),
                  initial_crack=is_crack,
                  is_displacement_boundary=simple_displacement_boundary)

    return model


@pytest.fixture(
    scope="session"
    )
def cl_model(data_path, request, simple_displacement_boundary):
    """Create a simple peridynamics Model object."""
    path = data_path
    mesh_file = path / "example_mesh.vtk"

    # Initiate integrator
    euler = EulerCL(dt=1e-3)

    # Create model
    model = Model(mesh_file, integrator=euler, horizon=0.1,
                  critical_stretch=0.005,
                  bond_stiffness=18.0 * 0.05 / (np.pi * 0.1**4),
                  initial_crack=is_crack,
                  is_displacement_boundary=simple_displacement_boundary)

    return model


@pytest.fixture(
    scope="session",
    params=[Euler, pytest.param(EulerCL, marks=context_available)]
    )
def simple_model(data_path, request, simple_displacement_boundary):
    """Create a simple peridynamics Model object."""
    path = data_path
    mesh_file = path / "example_mesh.vtk"

    # Initiate integrator
    euler = request.param(dt=1e-3)

    # Create model
    model = Model(mesh_file, integrator=euler, horizon=0.1,
                  critical_stretch=0.005,
                  bond_stiffness=18.0 * 0.05 / (np.pi * 0.1**4),
                  initial_crack=is_crack,
                  is_displacement_boundary=simple_displacement_boundary)

    return model

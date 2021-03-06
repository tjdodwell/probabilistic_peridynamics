"""Shared definitions for test modules."""
from ..cl import get_context
from ..model import Model, initial_crack_helper
from ..model_cl import ModelCL
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


@pytest.fixture(
    scope="session",
    params=[Model, pytest.param(ModelCL, marks=context_available)]
    )
def simple_model(data_path, request):
    """Create a simple peridynamics Model and ModelCL object."""
    path = data_path
    mesh_file = path / "example_mesh.vtk"

    @initial_crack_helper
    def is_crack(x, y):
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

    # Create model
    model = request.param(mesh_file, horizon=0.1, critical_strain=0.005,
                          elastic_modulus=0.05, initial_crack=is_crack)

    # Set left-hand side and right-hand side of boundary
    model.lhs = np.nonzero(model.coords[:, 0] < 1.5*model.horizon)
    model.rhs = np.nonzero(model.coords[:, 0] > 1.0 - 1.5*model.horizon)

    return model


@pytest.fixture(scope="session")
def simple_boundary_function():
    """Return a simple boundary function."""
    def boundary_function(model, u, step):
        load_rate = 0.00001

        u[model.lhs, 1:3] = 0.
        u[model.rhs, 1:3] = 0.

        extension = 0.5 * step * load_rate
        u[model.lhs, 0] = -extension
        u[model.rhs, 0] = extension

        return u
    return boundary_function

"""
A simple regression test.

A basic model is simulated for ten steps using the Euler integrator.
"""
from ..model import Model, initial_crack_helper
from ..integrators import Euler
import numpy as np
import pytest


@pytest.fixture(scope="module")
def simple_model(data_path):
    """Create a simple peridynamics Model object."""
    path = data_path
    mesh_file = path / "example_mesh.msh"

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
    model = Model(mesh_file, horizon=0.1, critical_strain=0.005,
                  elastic_modulus=0.05, initial_crack=is_crack)

    # Set left-hand side and right-hand side of boundary
    indices = np.arange(model.nnodes)
    model.lhs = indices[model.coords[:, 0] < 1.5*model.horizon]
    model.rhs = indices[model.coords[:, 0] > 1.0 - 1.5*model.horizon]

    return model


@pytest.fixture(scope="module")
def regression(simple_model):
    """Run the example simulation."""
    model = simple_model

    integrator = Euler(dt=1e-3)

    load_rate = 0.00001

    def boundary_function(model, u, step):
        u[model.lhs, 1:3] = np.zeros((len(model.lhs), 2))
        u[model.rhs, 1:3] = np.zeros((len(model.rhs), 2))

        u[model.lhs, 0] = (
            -0.5 * step * load_rate * np.ones(len(model.rhs))
            )
        u[model.rhs, 0] = (
            0.5 * step * load_rate * np.ones(len(model.rhs))
            )

        return u

    u, damage, *_ = model.simulate(
        steps=10,
        integrator=integrator,
        boundary_function=boundary_function
        )

    return model, u, damage


class TestRegression:
    """Regression tests."""

    def test_displacements(self, regression, data_path):
        """Ensure displacements are correct."""
        _, displacements, *_ = regression
        path = data_path

        expected_displacements = np.load(path/"expected_displacements.npy")
        assert np.allclose(displacements, expected_displacements)

    def test_damage(self, regression, data_path):
        """Ensure damage is correct."""
        _, _, damage = regression
        path = data_path

        expected_damage = np.load(path/"expected_damage.npy")
        assert np.all(damage == expected_damage)

    def test_mesh(self, regression, data_path, tmp_path):
        """Ensure mesh file is identical."""
        model, displacements, damage = regression
        path = data_path

        mesh = tmp_path / "mesh.vtk"
        model.write_mesh(mesh, damage, displacements)

        expected_mesh = path / "expected_mesh.vtk"

        assert (
            mesh.read_bytes().split(b"\n")[2:] ==
            expected_mesh.read_bytes().split(b"\n")[2:]
            )

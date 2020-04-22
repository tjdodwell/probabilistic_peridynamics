"""
A simple regression test.

A basic model is simulated for ten steps using the Euler integrator.
"""
from ..integrators import Euler
import numpy as np
import pytest


@pytest.fixture(scope="module")
def regression(simple_model, simple_boundary_function):
    """Run the example simulation."""
    model = simple_model
    boundary_function = simple_boundary_function

    integrator = Euler(dt=1e-3)

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
        assert np.allclose(damage, expected_damage)

    def test_mesh(self, regression, data_path, tmp_path):
        """Ensure mesh file is identical."""
        model, displacements, damage = regression
        path = data_path

        mesh = tmp_path / "mesh.vtk"
        model.write_mesh(mesh, damage, displacements)

        expected_mesh = path / "expected_mesh.vtk"

        assert(
            mesh.read_bytes().split(b"\n")[2:] ==
            expected_mesh.read_bytes().split(b"\n")[2:]
            )

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

    u, damage, connectivity, *_ = model.simulate(
        steps=10,
        integrator=integrator,
        boundary_function=boundary_function
        )

    return model, u, damage, connectivity


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
        _, _, damage, *_ = regression
        path = data_path

        expected_damage = np.load(path/"expected_damage.npy")
        assert np.allclose(damage, expected_damage)

    @pytest.mark.skip(reason="Stalling")
    def test_connectivity(self, regression, data_path):
        """Ensure connectivity is correct."""
        _, _, _, connectivity = regression
        path = data_path

        expected_connectivity = np.load(path/"expected_connectivity.npz")
        expected_nlist = expected_connectivity["nlist"]
        expected_n_neigh = expected_connectivity["n_neigh"]

        actual_nlist = connectivity[0]
        actual_n_neigh = connectivity[1]
        assert np.all(expected_nlist == actual_nlist)
        assert np.all(expected_n_neigh == actual_n_neigh)

    def test_mesh(self, regression, data_path, tmp_path):
        """Ensure mesh file is identical."""
        model, displacements, damage, *_ = regression
        path = data_path

        mesh = tmp_path / "mesh.vtk"
        model.write_mesh(mesh, damage, displacements)

        expected_mesh = path / "expected_mesh.vtk"

        assert(
            mesh.read_bytes().split(b"\n")[2:] ==
            expected_mesh.read_bytes().split(b"\n")[2:]
            )

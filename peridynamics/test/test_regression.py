"""
A simple regression test.

A basic model is simulated for ten steps using the Euler integrator.
"""
import numpy as np
import pytest


@pytest.fixture(scope="module")
def regression_cython(cython_model):
    """Run the example simulation."""
    model = cython_model
    steps = 10
    u, damage, connectivity, force, *_ = model.simulate(
        steps=steps,
        displacement_bc_magnitudes=0.00001 / 2 * np.linspace(
            1, steps, steps)
        )

    return model, u, damage, connectivity, force


@pytest.fixture(scope="module")
def regression_cl(cl_model):
    """Run the example simulation."""
    model = cl_model
    steps = 10
    u, damage, connectivity, force, *_ = model.simulate(
        steps=steps,
        displacement_bc_magnitudes=0.00001 / 2 * np.linspace(
            1, steps, steps)
        )

    return model, u, damage, connectivity, force


class TestRegression:
    """Regression tests."""

    def test_displacements_cython(self, regression_cython, data_path):
        """Ensure displacements are correct."""
        _, displacements, *_ = regression_cython
        path = data_path

        expected_displacements = np.load(path/"expected_displacements.npy")
        assert np.allclose(displacements, expected_displacements)

    def test_damage_cython(self, regression_cython, data_path):
        """Ensure damage is correct."""
        _, _, damage, *_ = regression_cython
        path = data_path
        expected_damage = np.load(path/"expected_damage.npy")
        assert np.allclose(damage, expected_damage)

    def test_connectivity_cython(self, regression_cython, data_path):
        """Ensure connectivity is correct."""
        _, _, _, connectivity, *_ = regression_cython
        path = data_path

        expected_connectivity = np.load(path/"expected_connectivity.npz")
        expected_nlist = expected_connectivity["nlist"]
        expected_n_neigh = expected_connectivity["n_neigh"]

        actual_nlist = connectivity[0]
        actual_n_neigh = connectivity[1]
        assert np.all(expected_nlist == actual_nlist)
        assert np.all(expected_n_neigh == actual_n_neigh)

    def test_force_cython(self, regression_cython, data_path):
        """Ensure connectivity is correct."""
        _, _, _, _, force = regression_cython
        path = data_path

        expected_force = np.load(path/"expected_force.npy")

        actual_force = force
        assert np.all(expected_force == actual_force)

    @pytest.mark.skip(reason="Stalling")
    def test_mesh_cython(self, regression_cython, data_path, tmp_path):
        """Ensure mesh file is identical."""
        model, displacements, damage, *_ = regression_cython
        path = data_path

        mesh = tmp_path / "mesh.vtk"
        model.write_mesh(mesh, damage, displacements)

        expected_mesh = path / "expected_mesh.vtk"

        assert(
            mesh.read_bytes().split(b"\n")[2:] ==
            expected_mesh.read_bytes().split(b"\n")[2:]
            )

    def test_connectivity_cl(self, regression_cl, data_path):
        """Ensure connectivity is correct."""
        _, _, _, connectivity, *_ = regression_cl
        path = data_path

        expected_connectivity = np.load(path/"expected_connectivity_cl.npz")
        expected_nlist = expected_connectivity["nlist"]
        expected_n_neigh = expected_connectivity["n_neigh"]

        actual_nlist = connectivity[0]
        actual_n_neigh = connectivity[1]
        assert np.all(expected_nlist == actual_nlist)
        assert np.all(expected_n_neigh == actual_n_neigh)

    def test_force_cl(self, regression_cl, data_path):
        """Ensure connectivity is correct."""
        _, _, _, _, force = regression_cl
        path = data_path

        expected_force = np.load(path/"expected_force.npy")

        actual_force = force
        assert np.allclose(expected_force, actual_force)

    @pytest.mark.skip(reason="Stalling")
    def test_mesh_cl(self, regression_cl, data_path, tmp_path):
        """Ensure mesh file is identical."""
        model, displacements, damage, *_ = regression_cl
        path = data_path

        mesh = tmp_path / "mesh.vtk"
        model.write_mesh(mesh, damage, displacements)

        expected_mesh = path / "expected_mesh_cl.vtk"

        assert(
            mesh.read_bytes().split(b"\n")[2:] ==
            expected_mesh.read_bytes().split(b"\n")[2:]
            )

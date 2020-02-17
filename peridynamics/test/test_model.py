"""
Tests for the model class
"""
from ..model import (Model, DimensionalityError, initial_crack_helper,
                     InvalidIntegrator)
import numpy as np
import pytest


@pytest.fixture(scope="module")
def basic_model_2d(data_path):
    mesh_file = data_path / "example_mesh.msh"
    model = Model(mesh_file, horizon=0.1, critical_strain=0.05,
                  elastic_modulus=0.05)
    return model


@pytest.fixture(scope="module")
def basic_model_3d(data_path):
    mesh_file = data_path / "example_mesh.msh"
    model = Model(mesh_file, horizon=0.1, critical_strain=0.05,
                  elastic_modulus=0.05, dimensions=3)
    return model


class TestDimension:
    def test_2d(self, basic_model_2d):
        model = basic_model_2d

        assert model.mesh_elements.connectivity == 'triangle'
        assert model.mesh_elements.boundary == 'line'

    @pytest.mark.skip(reason="No three dimensional example")
    def test_3d(self, basic_model_3d):
        model = basic_model_3d

        assert model.mesh_elements.connectivity == 'tetrahedron'
        assert model.mesh_elements.boundary == 'triangle'

    @pytest.mark.parametrize("dimensions", [1, 4])
    def test_dimensionality_error(self, dimensions):
        with pytest.raises(DimensionalityError):
            Model("abc.msh", horizon=0.1, critical_strain=0.05,
                  elastic_modulus=0.05, dimensions=dimensions)


class TestRead2D:
    """
    Test the _read_mesh method ensuring it correctly interprets the mesh file
    for a two dimensional system
    """
    def test_coords(self, basic_model_2d):
        model = basic_model_2d

        assert model.coords.shape == (2113, 3)
        assert model.nnodes == 2113
        assert np.all(
            model.coords[42] == np.array([1., 0.2499999999994083, 0.]))

    def test_mesh_connectivity(self, basic_model_2d):
        model = basic_model_2d

        assert model.mesh_connectivity.shape == (4096, 3)
        assert np.all(
            model.mesh_connectivity[100] == np.array([252, 651, 650]))

    def test_mesh_boundary(self, basic_model_2d):
        model = basic_model_2d

        assert model.mesh_boundary.shape == (128, 2)
        assert np.all(
            model.mesh_boundary[100] == np.array([100, 101]))


@pytest.mark.skip(reason="No three dimensional example")
class TestRead3D:
    """
    Test the _read_mesh method ensuring it correctly interprets the mesh file
    for a three dimensional system
    """
    def test_coords(self, basic_model_3d):
        assert 0

    def test_connectivity(self, basic_model_3d):
        assert 0

    def test_boundary_connectivity(self, basic_model_3d):
        assert 0


@pytest.fixture
def written_model(basic_model_3d, tmp_path):
    model = basic_model_3d
    mesh_file = tmp_path / "out_mesh.vtk"

    # Create synthetic damage and displacements
    # The damange is simply the number of the node (begining at 0) divided by
    # the total
    damage = np.arange(model.nnodes)/model.nnodes
    # The displacements are three columns of the damage array
    displacements = np.tile(damage, (3, 1)).T

    model.write_mesh(mesh_file, damage, displacements, file_format="vtk-ascii")
    model.write_mesh("abc.vtk", damage, displacements, file_format="vtk-ascii")

    return mesh_file


@pytest.mark.skip(reason="Should use a minimal example for testing this")
class TestWrite:
    """
    Tests for the writing of mesh files.
    """
    def test_coords(self, written_model):
        assert 0


class TestSimulate:
    """
    Tests for the simulate method.

    Further tests of simulation are in test_regression.py
    """
    def invalid_integrator(self, basic_model_2d):
        model = basic_model_2d
        with pytest.raises(InvalidIntegrator):
            model.simulate(10, None)


def test_initial_crack_helper():
    @initial_crack_helper
    def initial_crack(icoord, jcoord):
        critical_distance = 1.0
        if np.sum((jcoord - icoord)**2) > critical_distance:
            return True
        else:
            return False

    coords = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [5.0, 0.0, 0.0]
        ])

    actual = initial_crack(coords)
    expected = [
        (0, 3),
        (1, 2),
        (1, 3),
        (2, 3)
        ]

    assert expected == actual

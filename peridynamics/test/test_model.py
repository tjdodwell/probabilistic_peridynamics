"""
Tests for the model class
"""
from ..model import Model, DimensionalityError
import numpy as np
import pytest


@pytest.fixture(scope="module")
def basic_model_2d(data_path):
    model = Model()
    model.read_mesh(data_path / "example_mesh.msh")
    return model


@pytest.fixture(scope="module")
def basic_model_3d(data_path):
    model = Model()
    model.read_mesh(data_path / "example_mesh.msh")
    return model


class TestDimension:
    def test_2d(self):
        model = Model(dimensions=2)

        assert model.mesh_elements.connectivity == 'triangle'
        assert model.mesh_elements.boundary == 'line'

    def test_3d(self):
        model = Model(dimensions=3)

        assert model.mesh_elements.connectivity == 'tetrahedron'
        assert model.mesh_elements.boundary == 'triangle'

    @pytest.mark.parametrize("dimensions", [1, 4])
    def test_dimensionality_error(self, dimensions):
        with pytest.raises(DimensionalityError):
            Model(dimensions=dimensions)


class TestRead2D:
    """
    Test the read_mesh method ensuring it correctly interprets the mesh file
    for a two dimensional system
    """
    def test_coords(self, basic_model_2d):
        model = basic_model_2d

        assert model.coords.shape == (2113, 3)
        assert model.nnodes == 2113
        assert np.all(
            model.coords[42] == np.array([1., 0.2499999999994083, 0.]))

    def test_connectivity(self, basic_model_2d):
        model = basic_model_2d

        assert model.connectivity.shape == (4096, 3)
        assert model.nelem == 4096
        assert np.all(
            model.connectivity[100] == np.array([252, 651, 650]))

    def test_boundary_connectivity(self, basic_model_2d):
        model = basic_model_2d

        assert model.connectivity_bnd.shape == (128, 2)
        assert model.nelem_bnd == 128
        assert np.all(
            model.connectivity_bnd[100] == np.array([100, 101]))


@pytest.mark.skip(reason="No three dimensional example")
class TestRead3D:
    """
    Test the read_mesh method ensuring it correctly interprets the mesh file
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

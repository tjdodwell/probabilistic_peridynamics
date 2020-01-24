"""
Tests for the reading and writing of mesh data
"""
from ..SeqPeriVectorized import SeqModel as Model
import numpy as np
import pathlib
import pytest


@pytest.fixture
def basic_model_2d():
    mesh_file = (
        pathlib.Path(__file__).parent.absolute() / "data/example_mesh.msh")
    model = Model()
    model.read_mesh(mesh_file)

    return model


@pytest.fixture
def basic_model_3d():
    mesh_file = (
        pathlib.Path(__file__).parent.absolute() / "data/example_mesh.msh")
    model = Model()
    model.read_mesh(mesh_file)

    return model


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

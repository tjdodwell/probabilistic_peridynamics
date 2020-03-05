"""Tests for the model class."""
from ..model import (Model, DimensionalityError, initial_crack_helper,
                     InvalidIntegrator)
import numpy as np
import scipy.sparse as sparse
import pytest


@pytest.fixture(scope="module")
def basic_model_2d(data_path):
    """Create a basic 2D model object."""
    mesh_file = data_path / "example_mesh.msh"
    model = Model(mesh_file, horizon=0.1, critical_strain=0.05,
                  elastic_modulus=0.05)
    return model


@pytest.fixture(scope="module")
def basic_model_3d(data_path):
    """Create a basic 3D model object."""
    mesh_file = data_path / "example_mesh.msh"
    model = Model(mesh_file, horizon=0.1, critical_strain=0.05,
                  elastic_modulus=0.05, dimensions=3)
    return model


class TestDimension:
    """Test the dimension argument of the Model class."""

    def test_2d(self, basic_model_2d):
        """Test initialisation of a 2D model."""
        model = basic_model_2d

        assert model.mesh_elements.connectivity == 'triangle'
        assert model.mesh_elements.boundary == 'line'

    @pytest.mark.skip(reason="No three dimensional example")
    def test_3d(self, basic_model_3d):
        """Test initialisation of a 3D model."""
        model = basic_model_3d

        assert model.mesh_elements.connectivity == 'tetrahedron'
        assert model.mesh_elements.boundary == 'triangle'

    @pytest.mark.parametrize("dimensions", [1, 4])
    def test_dimensionality_error(self, dimensions):
        """Test invalid dimension arguments."""
        with pytest.raises(DimensionalityError):
            Model("abc.msh", horizon=0.1, critical_strain=0.05,
                  elastic_modulus=0.05, dimensions=dimensions)


class TestRead2D:
    """Test reading a mesh with a 2D model."""

    def test_coords(self, basic_model_2d):
        """Test coordinates are read correctly."""
        model = basic_model_2d

        assert model.coords.shape == (2113, 3)
        assert model.nnodes == 2113
        assert np.all(
            model.coords[42] == np.array([1., 0.2499999999994083, 0.]))

    def test_mesh_connectivity(self, basic_model_2d):
        """Test mesh connectivity is read correctly."""
        model = basic_model_2d

        assert model.mesh_connectivity.shape == (4096, 3)
        assert np.all(
            model.mesh_connectivity[100] == np.array([252, 651, 650]))

    def test_mesh_boundary(self, basic_model_2d):
        """Test mesh boundary is read correctly."""
        model = basic_model_2d

        assert model.mesh_boundary.shape == (128, 2)
        assert np.all(
            model.mesh_boundary[100] == np.array([100, 101]))


@pytest.mark.skip(reason="No three dimensional example")
class TestRead3D:
    """Test reading a mesh with a 3D model."""

    def test_coords(self, basic_model_3d):
        """Test coordinates are read correctly."""
        assert 0

    def test_connectivity(self, basic_model_3d):
        """Test mesh connectivity is read correctly."""
        assert 0

    def test_boundary_connectivity(self, basic_model_3d):
        """Test mesh boundary is read correctly."""
        assert 0


@pytest.fixture
def written_model(basic_model_3d, tmp_path):
    """Write an example mesh file from a model."""
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
    """Tests for the writing of mesh files."""

    def test_coords(self, written_model):
        """Ensure coordinates are written correctly."""
        assert 0


def test_volume_2d(basic_model_2d, data_path):
    """Test volume calculation."""
    expected_volume = np.load(data_path/"expected_volume.npy")
    assert np.all(basic_model_2d.volume == expected_volume)


def test_bond_stiffness_2d(basic_model_2d):
    """Test bond stiffness calculation."""
    assert np.isclose(basic_model_2d.bond_stiffness, 2864.7889756)


def test_neighbourhood(basic_model_2d, data_path):
    """Test _neighbourhood method."""
    expected_neighbourhood = sparse.load_npz(
        data_path/"expected_neighbourhood.npz"
        )
    assert np.all(
        ~(basic_model_2d._neighbourhood() != expected_neighbourhood).toarray()
        )


class TestConnectivity:
    """Test the _connectivity method."""

    def test_basic_connectivity(self, basic_model_2d, data_path):
        """Test connectivity calculation with no initial crack."""
        expected_connectivity = sparse.load_npz(
            data_path/"expected_connectivity_basic.npz"
            )
        assert np.all(
            ~(
                basic_model_2d.initial_connectivity != expected_connectivity
                ).toarray()
            )

    def test_connectivity(self, simple_model, data_path):
        """Test connectivity calculation with no initial crack."""
        expected_connectivity = sparse.load_npz(
            data_path/"expected_connectivity_crack.npz"
            )
        assert np.all(
            ~(
                simple_model.initial_connectivity != expected_connectivity
                ).toarray()
            )


def test_displacements():
    """Test displacement calculation."""
    r = np.identity(3, dtype=np.float)
    expected_d_x = np.array([[0.0, 1.0, 1.0],
                             [-1.0, 0.0, 0.0],
                             [-1.0, 0.0, 0.0]])
    expected_d_y = np.array([[0.0, -1.0, 0.0],
                             [1.0, 0.0, 1.0],
                             [0.0, -1.0, 0.0]])
    expected_d_z = np.array([[0.0, 0.0, -1.0],
                             [0.0, 0.0, -1.0],
                             [1.0, 1.0, 0.0]])

    d_x, d_y, d_z = Model._displacements(r)

    assert np.all(d_x == expected_d_x)
    assert np.all(d_y == expected_d_y)
    assert np.all(d_z == expected_d_z)


class TestSimulate:
    """
    Tests for the simulate method.

    Further tests of simulation are in test_regression.py
    """

    def invalid_integrator(self, basic_model_2d):
        """Test passing an invalid integrator to simulate."""
        model = basic_model_2d
        with pytest.raises(InvalidIntegrator):
            model.simulate(10, None)


class TestInitialCrackHelper:
    """Tests of the initial crack helper decorator."""

    coords = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [5.0, 0.0, 0.0]
        ])

    def test_initial_crack_helper(self):
        """Test with all particles interacting."""
        @initial_crack_helper
        def initial_crack(icoord, jcoord):
            critical_distance = 1.0
            if np.sum((jcoord - icoord)**2) > critical_distance:
                return True
            else:
                return False

        actual = initial_crack(self.coords,
                               sparse.csr_matrix(np.ones((4, 4), dtype=bool)))
        expected = [
            (0, 3),
            (1, 2),
            (1, 3),
            (2, 3)
            ]

        assert expected == actual

    def test_neighbourhood(self):
        """Test with a neighbourhood defined."""
        @initial_crack_helper
        def initial_crack(icoord, jcoord):
            critical_distance = 1.0
            if np.sum((jcoord - icoord)**2) > critical_distance:
                return True
            else:
                return False

        # Create a neighbourhood matrix, ensure that particle 3 is not in the
        # neighbourhood of any other nodes
        neighbourhood = np.zeros((4, 4), dtype=bool)
        neighbourhood[0, 1] = True
        neighbourhood[0, 2] = True
        neighbourhood[1, 2] = True
        neighbourhood = neighbourhood + neighbourhood.T
        neighbourhood = sparse.csr_matrix(neighbourhood)

        actual = initial_crack(self.coords, neighbourhood)
        expected = [
            (1, 2)
            ]

        assert expected == actual

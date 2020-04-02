"""Tests for the model class."""
from ..model import (Model, DimensionalityError, initial_crack_helper,
                     InvalidIntegrator)
from ..integrators import Euler
import meshio
import numpy as np
import scipy.sparse as sparse
import pytest


@pytest.fixture(scope="module")
def basic_model_2d(data_path):
    """Create a basic 2D model object."""
    mesh_file = data_path / "example_mesh.vtk"
    model = Model(mesh_file, horizon=0.1, critical_strain=0.05,
                  elastic_modulus=0.05)
    return model


@pytest.fixture(scope="module")
def basic_model_3d(data_path):
    """Create a basic 3D model object."""
    mesh_file = data_path / "example_mesh_3d.vtk"
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

    def test_3d(self, basic_model_3d):
        """Test initialisation of a 3D model."""
        model = basic_model_3d

        assert model.mesh_elements.connectivity == 'tetra'
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
            model.coords[42] == np.array([1., 0.2499999999994083, 0.])
            )

    def test_mesh_connectivity(self, basic_model_2d):
        """Test mesh connectivity is read correctly."""
        model = basic_model_2d

        assert model.mesh_connectivity.shape == (4096, 3)
        assert np.all(
            model.mesh_connectivity[100] == np.array([252, 651, 650])
            )

    def test_mesh_boundary(self, basic_model_2d):
        """Test mesh boundary is read correctly."""
        model = basic_model_2d

        assert model.mesh_boundary.shape == (128, 2)
        assert np.all(model.mesh_boundary[100] == np.array([100, 101]))


class TestRead3D:
    """Test reading a mesh with a 3D model."""

    def test_coords(self, basic_model_3d):
        """Test coordinates are read correctly."""
        model = basic_model_3d

        assert model.coords.shape == (1023, 3)
        assert model.nnodes == 1023
        assert np.allclose(model.coords[42], np.array([1., 0.2, 0.]))

    def test_connectivity(self, basic_model_3d):
        """Test mesh connectivity is read correctly."""
        model = basic_model_3d

        assert model.mesh_connectivity.shape == (3948, 4)
        assert np.all(
            model.mesh_connectivity[100] == np.array([467, 733, 802, 937])
            )

    def test_boundary_connectivity(self, basic_model_3d):
        """Test mesh boundary is read correctly."""
        model = basic_model_3d

        assert model.mesh_boundary.shape == (1408, 3)
        assert np.all(model.mesh_boundary[100] == np.array([151, 122, 162]))


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


@pytest.fixture(scope="class")
def written_example(basic_model_3d, tmp_path_factory):
    """Write an example model to a mesh object."""
    model = basic_model_3d

    damage = np.random.random(model.nnodes)
    u = np.random.random((model.nnodes, 3))

    mesh_file = tmp_path_factory.mktemp("data")/"mesh.vtk"
    model.write_mesh(mesh_file, damage=damage, displacements=u)

    mesh = meshio.read(mesh_file)

    return model, mesh, u, damage


class TestWrite:
    """Tests for the writing of mesh files."""

    def test_coords(self, written_example):
        """Ensure coordinates are written correctly."""
        model, mesh, u, damage = written_example
        assert np.all(model.coords == mesh.points)

    def test_mesh_connectivity(self, written_example):
        """Ensure connectivity is written correctly."""
        model, mesh, u, damage = written_example
        assert np.all(
            model.mesh_connectivity == mesh.cells_dict[
                model.mesh_elements.connectivity
                ]
            )

    def test_mesh_boundary(self, written_example):
        """Ensure boundary is written correctly."""
        model, mesh, u, damage = written_example
        assert np.all(
            model.mesh_boundary == mesh.cells_dict[
                model.mesh_elements.boundary
                ]
            )

    def test_damage(self, written_example):
        """Ensure damage is written correctly."""
        model, mesh, u, damage = written_example
        assert np.all(damage == mesh.point_data["damage"])

    def test_displacements(self, written_example):
        """Ensure displacements are written correctly."""
        model, mesh, u, damage = written_example
        assert np.all(u == mesh.point_data["displacements"])


def test_volume_2d(basic_model_2d, data_path):
    """Test volume calculation."""
    expected_volume = np.load(data_path/"expected_volume.npy")
    assert np.allclose(basic_model_2d.volume, expected_volume)


def test_volume_3d(basic_model_3d, data_path):
    """Test volume calculation."""
    expected_volume = np.load(data_path/"expected_volume_3d.npy")
    assert np.allclose(basic_model_3d.volume, expected_volume)


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
    expected_d_x = np.array([[0.0, -1.0, -1.0],
                             [1.0, 0.0, 0.0],
                             [1.0, 0.0, 0.0]])
    expected_d_y = np.array([[0.0, 1.0, 0.0],
                             [-1.0, 0.0, -1.0],
                             [0.0, 1.0, 0.0]])
    expected_d_z = np.array([[0.0, 0.0, 1.0],
                             [0.0, 0.0, 1.0],
                             [-1.0, -1.0, 0.0]])

    d_x, d_y, d_z = Model._displacements(r)

    assert np.all(d_x == expected_d_x)
    assert np.all(d_y == expected_d_y)
    assert np.all(d_z == expected_d_z)


@pytest.fixture(scope="module")
def model_force_test(data_path):
    """Create a minimal model designed for testings force calculation."""
    mesh_file = data_path/"force_test.vtk"
    model = Model(mesh_file, horizon=1.01, critical_strain=0.05,
                  elastic_modulus=0.05)
    return model


class TestForce():
    """Test force calculation."""

    def test_initial_force(self, model_force_test):
        """Ensure initial forces are zero."""
        model = model_force_test
        connectivity = model.initial_connectivity

        H_x, H_y, H_z, L = model._H_and_L(model.coords, connectivity)
        strain = model._strain(L)
        f = model._bond_force(strain, connectivity, L, H_x, H_y, H_z)

        assert np.all(f == 0)

    def test_force(self, model_force_test):
        """Ensure forces are in the correct direction using a minimal model."""
        model = model_force_test
        connectivity = model.initial_connectivity

        # Nodes 0 and 1 are connected along the x axis, 1 and 2 along the y
        # axis. There are no other connections.
        assert connectivity[1, 0]
        assert not connectivity[2, 0]
        assert connectivity[2, 1]

        # Displace nodes 1 and 2 in the positive x direction and y in the
        # positive y direction
        u = np.array([
            [0.0, 0.0, 0.0],
            [0.05, 0.0, 0.0],
            [0.05, 0.05, 0.0]
            ])

        # Calculate force
        # This is lifted from the Model.simulate method
        H_x, H_y, H_z, L = model._H_and_L(model.coords+u, connectivity)
        strain = model._strain(L)
        f = model._bond_force(strain, connectivity, L, H_x, H_y, H_z)

        # Ensure force array is correct
        force_value = 0.00229417
        expected_force = np.array([
            [force_value, 0., 0.],
            [-force_value, force_value, 0.],
            [0., -force_value, 0.]
            ])
        assert np.allclose(f, expected_force)

        # Ensure force is restorative,
        #   - Node 1 pulls node 0 in the positive x direction
        #   - Node 0 pulls node 1 in the negative x direction
        assert f[0, 0] > 0
        assert f[1, 0] < 0
        #   - Node 2 pulls node 1 in the positive y direction
        #   - Node 1 pulls node 2 in the negative y direction
        assert f[1, 1] > 0
        assert f[2, 1] < 0

        # Node 0 has no component of force in the y or z dimensions
        assert np.all(f[0, 1:] == 0)
        # Node 1 has no component of force in the z dimension
        assert f[1, 2] == 0
        # Node 2 has no component of force in the x or z dimensions
        assert np.all(f[2, [0, 2]] == 0)


class TestSimulate:
    """
    Tests for the simulate method.

    Further tests of simulation are in test_regression.py
    """

    def test_invalid_integrator(self, basic_model_2d):
        """Test passing an invalid integrator to simulate."""
        model = basic_model_2d

        with pytest.raises(InvalidIntegrator):
            model.simulate(10, None)

    def test_stateless(self, simple_model, simple_boundary_function):
        """Ensure the simulate method does not affect the state of Models."""
        model = simple_model
        euler = Euler(dt=1e-3)

        u, damage, connectivity = model.simulate(
            steps=2,
            integrator=euler,
            boundary_function=simple_boundary_function
            )

        expected_u, expected_damage, expected_connectivity = model.simulate(
            steps=2,
            integrator=euler,
            boundary_function=simple_boundary_function
            )

        assert np.all(u == expected_u)
        assert np.all(damage == expected_damage)
        assert np.all(
            connectivity.toarray() == expected_connectivity.toarray()
            )

    def test_restart(self, simple_model, simple_boundary_function):
        """Ensure simulation restarting gives consistent results."""
        model = simple_model
        euler = Euler(dt=1e-3)

        u, damage, connectivity = model.simulate(
            steps=1,
            integrator=euler,
            boundary_function=simple_boundary_function
            )
        u, damage, connectivity = model.simulate(
            steps=1,
            integrator=euler,
            boundary_function=simple_boundary_function,
            u=u,
            connectivity=connectivity,
            first_step=2
            )

        expected_u, expected_damage, expected_connectivity = model.simulate(
            steps=2,
            integrator=euler,
            boundary_function=simple_boundary_function
            )

        assert np.all(u == expected_u)
        assert np.all(damage == expected_damage)
        assert np.all(
            connectivity.toarray() == expected_connectivity.toarray()
            )

    def test_restart_dense(self, simple_model, simple_boundary_function):
        """Ensure simulation restarting works with dense connectivity."""
        model = simple_model
        euler = Euler(dt=1e-3)

        u, damage, connectivity = model.simulate(
            steps=1,
            integrator=euler,
            boundary_function=simple_boundary_function
            )
        u, damage, connectivity = model.simulate(
            steps=1,
            integrator=euler,
            boundary_function=simple_boundary_function,
            u=u,
            connectivity=connectivity.toarray(),
            first_step=2
            )

        expected_u, expected_damage, expected_connectivity = model.simulate(
            steps=2,
            integrator=euler,
            boundary_function=simple_boundary_function
            )

        assert np.all(u == expected_u)
        assert np.all(damage == expected_damage)
        assert np.all(
            connectivity.toarray() == expected_connectivity.toarray()
            )

    def test_no_boundary_function(self, simple_model):
        """Ensure passing no boundary function works correctly."""
        model = simple_model
        euler = Euler(dt=1e-3)

        u, damage, connectivity = model.simulate(
            steps=2,
            integrator=euler,
            boundary_function=None
            )

        def boundary_function(model, u, step):
            return u

        expected_u, expected_damage, expected_connectivity = model.simulate(
            steps=2,
            integrator=euler,
            boundary_function=boundary_function
            )

        assert np.all(u == expected_u)
        assert np.all(damage == expected_damage)
        assert np.all(
            connectivity.toarray() == expected_connectivity.toarray()
            )

    def test_write(self, simple_model, simple_boundary_function, tmp_path):
        """Ensure that the mesh file written by simulate is correct."""
        model = simple_model
        euler = Euler(dt=1e-3)

        u, damage, connectivity = model.simulate(
            steps=1,
            integrator=euler,
            boundary_function=simple_boundary_function,
            write=1,
            write_path=tmp_path
            )

        mesh = tmp_path / "U_1.vtk"

        expected_mesh = tmp_path / "mesh.vtk"
        model.write_mesh(expected_mesh, damage, u)

        assert mesh.read_bytes() == expected_mesh.read_bytes()


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

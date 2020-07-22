"""Tests for the model class."""
from .conftest import context_available
from ..model import (Model, DimensionalityError, FamilyError,
                     initial_crack_helper, InvalidIntegrator)
from ..integrators import Euler, EulerOpenCL
import meshio
import numpy as np
import pytest


@pytest.fixture(
    scope="session",
    params=[Euler, pytest.param(EulerOpenCL, marks=context_available)])
def basic_models_2d(data_path, request, simple_displacement_boundary):
    """Create a basic 2D model object."""

    mesh_file = data_path / "example_mesh.vtk"
    euler = request.param(dt=1e-3)
    model = Model(mesh_file, integrator=euler, horizon=0.1,
                  critical_stretch=0.05,
                  bond_stiffness=18.0 * 0.05 / (np.pi * 0.1**4),
                  is_displacement_boundary=simple_displacement_boundary)
    return model


@pytest.fixture()
def basic_model_2d(data_path, simple_displacement_boundary):
    """Create a basic 2D model object."""
    mesh_file = data_path / "example_mesh.vtk"

    euler = Euler(dt=1e-3)
    model = Model(mesh_file, integrator=euler, horizon=0.1,
                  critical_stretch=0.05,
                  bond_stiffness=18.0 * 0.05 / (np.pi * 0.1**4),
                  is_displacement_boundary=simple_displacement_boundary)
    return model


@pytest.fixture(
    scope="session",
    params=[Euler, pytest.param(EulerOpenCL, marks=context_available)])
def basic_models_3d(data_path, simple_displacement_boundary, request):
    """Create a basic 2D model object."""
    mesh_file = data_path / "example_mesh_3d.vtk"
    euler = request.param(dt=1e-3)
    model = Model(mesh_file, integrator=euler, horizon=0.1,
                  critical_stretch=0.05,
                  bond_stiffness=18.0 * 0.05 / (np.pi * 0.1**4),
                  dimensions=3,
                  is_displacement_boundary=simple_displacement_boundary)
    return model


@pytest.fixture()
def basic_model_3d(data_path, simple_displacement_boundary):
    """Create a basic 3D model object."""
    mesh_file = data_path / "example_mesh_3d.vtk"
    euler = Euler(dt=1e-3)
    model = Model(mesh_file, integrator=euler, horizon=0.1,
                  critical_stretch=0.05,
                  bond_stiffness=18.0 * 0.05 / (np.pi * 0.1**4),
                  dimensions=3,
                  is_displacement_boundary=simple_displacement_boundary)
    return model


class TestDimension:
    """Test the dimension argument of the Model class."""

    def test_2d(self, basic_models_2d):
        """Test initialisation of a 2D model."""
        model = basic_models_2d

        assert model.mesh_elements.connectivity == 'triangle'
        assert model.mesh_elements.boundary == 'line'

    def test_3d(self, basic_models_3d):
        """Test initialisation of a 3D model."""
        model = basic_models_3d

        assert model.mesh_elements.connectivity == 'tetra'
        assert model.mesh_elements.boundary == 'triangle'

    @pytest.mark.parametrize("dimensions", [1, 4])
    def test_dimensionality_error(self, dimensions):
        """Test invalid dimension arguments."""
        integrator = Euler(dt=1e-3)
        with pytest.raises(DimensionalityError) as exception:
            Model("abc.msh", integrator=integrator,
                  horizon=0.1, critical_stretch=0.05,
                  bond_stiffness=18.0 * 0.05 / (np.pi * 0.1**4),
                  dimensions=dimensions)
            assert str(dimensions) in exception.value


class TestRead2D:
    """Test reading a mesh with a 2D model."""

    def test_coords(self, basic_models_2d):
        """Test coordinates are read correctly."""
        model = basic_models_2d

        assert model.coords.shape == (2113, 3)
        assert model.nnodes == 2113
        assert np.all(
            model.coords[42] == np.array([1., 0.2499999999994083, 0.])
            )

    def test_mesh_connectivity(self, basic_models_2d):
        """Test mesh connectivity is read correctly."""
        model = basic_models_2d

        assert model.mesh_connectivity.shape == (4096, 3)
        assert np.all(
            model.mesh_connectivity[100] == np.array([252, 651, 650])
            )

    def test_mesh_boundary(self, basic_models_2d):
        """Test mesh boundary is read correctly."""
        model = basic_models_2d

        assert model.mesh_boundary.shape == (128, 2)
        assert np.all(model.mesh_boundary[100] == np.array([100, 101]))


class TestRead3D:
    """Test reading a mesh with a 3D model."""

    def test_coords(self, basic_models_3d):
        """Test coordinates are read correctly."""
        model = basic_models_3d

        assert model.coords.shape == (1175, 3)
        assert model.nnodes == 1175
        assert np.allclose(model.coords[42], np.array([0.5, 0.1, 0.]))

    def test_connectivity(self, basic_models_3d):
        """Test mesh connectivity is read correctly."""
        model = basic_models_3d

        assert model.mesh_connectivity.shape == (4788, 4)
        assert np.all(
            model.mesh_connectivity[100] == np.array([833, 841, 817, 1168])
            )

    def test_boundary_connectivity(self, basic_models_3d):
        """Test mesh boundary is read correctly."""
        model = basic_models_3d

        assert model.mesh_boundary.shape == (1474, 3)
        assert np.all(model.mesh_boundary[100] == np.array([172, 185, 124]))


@pytest.fixture(scope="class")
def written_example(basic_models_3d, tmp_path_factory):
    """Write an example model to a mesh object."""
    model = basic_models_3d

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


def test_volume_2d(basic_models_2d, data_path):
    """Test volume calculation."""
    expected_volume = np.load(data_path/"expected_volume.npy")
    assert np.allclose(basic_models_2d.volume, expected_volume)


def test_volume_3d(basic_models_3d, data_path):
    """Test volume calculation."""
    expected_volume = np.load(data_path/"expected_volume_3d.npy")
    assert np.allclose(basic_models_3d.volume, expected_volume)


def test_bond_stiffness_2d(basic_models_2d):
    """Test bond stiffness calculation."""
    assert np.isclose(basic_models_2d.bond_stiffness, 2864.7889756)


class TestConnectivity:
    """Test the initial neighbour list."""

    def test_basic_connectivity(self, basic_models_2d, data_path):
        """Test connectivity calculation with no initial crack."""
        npz_file = np.load(
            data_path/"expected_connectivity_basic.npz"
            )
        expected_nlist = npz_file["nlist"]
        expected_n_neigh = npz_file["n_neigh"]

        actual_nlist, actual_n_neigh = basic_models_2d.initial_connectivity
        assert np.all(expected_nlist == actual_nlist)
        assert np.all(expected_n_neigh == actual_n_neigh)

    def test_connectivity(self, simple_model, data_path):
        """Test connectivity calculation with initial crack."""
        npz_file = np.load(
            data_path/"expected_connectivity_crack.npz"
            )
        expected_nlist = npz_file["nlist"]
        expected_n_neigh = npz_file["n_neigh"]

        actual_nlist, actual_n_neigh = simple_model.initial_connectivity
        assert np.all(expected_nlist == actual_nlist)
        assert np.all(expected_n_neigh == actual_n_neigh)


def test_initial_damage_2d(basic_model_2d):
    """Ensure initial damage is zero."""
    model = basic_model_2d
    connectivity = model.initial_connectivity
    damage = model._damage(connectivity[1])

    assert np.all(damage == 0)


def test_initial_damage_3d(basic_model_3d):
    """Ensure initial damage is zero."""
    model = basic_model_3d
    connectivity = model.initial_connectivity
    damage = model._damage(connectivity[1])

    assert np.all(damage == 0)


def test_family_error(data_path):
    """Test raising of exception when a node has no neighbours."""
    with pytest.raises(FamilyError):
        mesh_file = data_path / "example_mesh_3d.vtk"
        Model(mesh_file, horizon=0.0001, critical_stretch=0.05,
              bond_stiffness=18.0 * 0.05 / (np.pi * 0.0001**4),
              dimensions=3)


@pytest.fixture(scope="module")
def model_force_test(data_path):
    """Create a minimal model designed for testings force calculation."""
    mesh_file = data_path/"force_test.vtk"
    model = Model(mesh_file, horizon=1.01, critical_stretch=0.05,
                  bond_stiffness=18.0 * 0.05 / (np.pi * 1.01**4))
    return model


class TestForce():
    """Test force calculation."""

    def test_initial_force(self, model_force_test):
        """Ensure initial forces are zero."""
        model = model_force_test
        nlist, n_neigh = model.initial_connectivity

        u = np.zeros_like(model.coords)
        f = model._bond_force(u, nlist, n_neigh)

        assert np.all(f == 0)

    def test_force(self, model_force_test):
        """Ensure forces are in the correct direction using a minimal model."""
        model = model_force_test
        nlist, n_neigh = model.initial_connectivity

        # Nodes 0 and 1 are connected along the x axis, 1 and 2 along the y
        # axis. There are no other connections.
        assert n_neigh[0] == 1
        assert n_neigh[1] == 2
        assert n_neigh[2] == 1
        assert 1 in nlist[0]
        assert 0 in nlist[1]
        assert 2 in nlist[1]
        assert 1 in nlist[2]

        # Displace nodes 1 and 2 in the positive x direction and y in the
        # positive y direction
        u = np.array([
            [0.0, 0.0, 0.0],
            [0.05, 0.0, 0.0],
            [0.05, 0.05, 0.0]
            ])

        # Calculate force
        # This is lifted from the Model.simulate method
        f = model._bond_force(u, nlist, n_neigh)

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


class TestBoundaryConditions:
    """Tests for the _set_boundary_conditions method."""

    @pytest.fixture(
        scope="module",
        params=[Euler, pytest.param(EulerOpenCL, marks=context_available)])
    def test_invalid_boundary_function(self, data_path, request):
        mesh_file = data_path / "example_mesh.vtk"
        euler = request.param(dt=1e-3)
        invalid_boundary_function = [None, None, None]
        with pytest.raises(TypeError) as exception:
            model = Model(
                mesh_file, integrator=euler, horizon=0.1,
                critical_stretch=0.05,
                bond_stiffness=18.0 * 0.05 / (np.pi * 0.1**4),
                is_displacement_boundary=invalid_boundary_function)
            assert("is_displacement_boundary must be a *function*."
                   in exception.value)

    @pytest.fixture(
        scope="module",
        params=[Euler, pytest.param(EulerOpenCL, marks=context_available)])
    def test_invalid_boundary_function2(self, data_path, request):
        mesh_file = data_path / "example_mesh.vtk"
        euler = request.param(dt=1e-3)

        def invalid_boundary_function():
            """Return an invalid boundary function."""
            return 1

        with pytest.raises(TypeError) as exception:
            model = Model(
                mesh_file, integrator=euler, horizon=0.1,
                critical_stretch=0.05,
                bond_stiffness=18.0 * 0.05 / (np.pi * 0.1**4),
                is_displacement_boundary=invalid_boundary_function)
            assert("is_displacement_boundary must be a function that returns"
                   + " a *list*." in exception.value)

    @pytest.fixture(
        scope="module",
        params=[Euler, pytest.param(EulerOpenCL, marks=context_available)])
    def test_invalid_boundary_function3(self, data_path, request):
        mesh_file = data_path / "example_mesh.vtk"
        euler = request.param(dt=1e-3)

        def invalid_boundary_function():
            """Return an invalid boundary function."""
            return [None]

        with pytest.raises(TypeError) as exception:
            model = Model(
                mesh_file, integrator=euler, horizon=0.1,
                critical_stretch=0.05,
                bond_stiffness=18.0 * 0.05 / (np.pi * 0.1**4),
                is_displacement_boundary=invalid_boundary_function)
            assert("{} must return a function that returns a list"
                   + " of length *3* of floats or None" in exception.value)

    @pytest.fixture(
        scope="module",
        params=[Euler, pytest.param(EulerOpenCL, marks=context_available)])
    def test_no_boundary_function(self, data_path, request):
        """Ensure passing no boundary function works correctly."""
        mesh_file = data_path / "example_mesh.vtk"
        euler = request.param(dt=1e-3)

        def is_displacement_boundary(x):
            return [None, None, None]

        model = Model(
            mesh_file, integrator=euler, horizon=0.1,
            critical_stretch=0.05,
            bond_stiffness=18.0 * 0.05 / (np.pi * 0.1**4),
            is_displacement_boundary=is_displacement_boundary)

        u, damage, connectivity = model.simulate(
            steps=2,
            max_displacement_rate=0.000005/2
            )

        model = Model(
            mesh_file, integrator=euler, horizon=0.1,
            critical_stretch=0.05,
            bond_stiffness=18.0 * 0.05 / (np.pi * 0.1**4))

        expected_u, expected_damage, expected_connectivity = model.simulate(
            steps=2,
            max_displacement_rate=0.000005/2
            )

        assert np.all(u == expected_u)
        assert np.all(damage == expected_damage)
        assert np.all(connectivity[0] == expected_connectivity[0])
        assert np.all(connectivity[1] == expected_connectivity[1])


class TestIntegrator:
    """ Tests for the simulate method.

    Further tests of integrator are in test_integrator.py
    """
    @pytest.fixture(scope="module")
    def test_invalid_integrator(self, data_path):
        """Test passing an invalid integrator to simulate."""
        mesh_file = data_path / "example_mesh.vtk"
        invalid_integrator = None
        with pytest.raises(InvalidIntegrator):
            model = Model(
                mesh_file, integrator=invalid_integrator, horizon=0.1,
                critical_stretch=0.05,
                bond_stiffness=18.0 * 0.05 / (np.pi * 0.1**4))


class TestSimulate:
    """
    Tests for the simulate method.

    Further tests of simulation are in test_regression.py
    """

    def test_invalid_connectivity(self, basic_models_2d):
        """Test passing an invalid connectivity argument to simulate."""
        with pytest.raises(TypeError) as exception:
            basic_models_2d.simulate(steps=10, connectivity=[1, 2, 3])
            assert "connectivity must be a tuple or None" in exception.value

    def test_invalid_connectivity2(self, basic_models_2d):
        """Test passing an invalid connectivity argument to simulate."""
        with pytest.raises(ValueError) as exception:
            basic_models_2d.simulate(10, connectivity=(1, 2, 3))
            assert "connectivity must be of size 2" in exception.value

    def test_stateless(self, simple_model, simple_boundary_function):
        """Ensure the simulate method does not affect the state of Models."""
        model = simple_model

        u, damage, connectivity = model.simulate(
            steps=2,
            max_displacement_rate=0.000005/2)

        expected_u, expected_damage, expected_connectivity = model.simulate(
            steps=2,
            max_displacement_rate=0.000005/2)

        assert np.all(u == expected_u)
        assert np.all(damage == expected_damage)
        assert np.all(connectivity[0] == expected_connectivity[0])
        assert np.all(connectivity[1] == expected_connectivity[1])

    def test_restart(self, simple_model, simple_boundary_function):
        """Ensure simulation restarting gives consistent results."""
        model = simple_model

        u, damage, connectivity = model.simulate(
            steps=1,
            max_displacement_rate=0.000005/2
            )
        u, damage, connectivity = model.simulate(
            steps=1,
            max_displacement_rate=0.000005/2,
            u=u,
            connectivity=connectivity,
            first_step=2
            )

        expected_u, expected_damage, expected_connectivity = model.simulate(
            steps=2,
            max_displacement_rate=0.000005/2
            )

        assert np.all(u == expected_u)
        assert np.all(damage == expected_damage)
        assert np.all(connectivity[0] == expected_connectivity[0])
        assert np.all(connectivity[1] == expected_connectivity[1])

    def test_write(self, simple_model, simple_boundary_function, tmp_path):
        """Ensure that the mesh file written by simulate is correct."""
        model = simple_model

        u, damage, connectivity = model.simulate(
            steps=1,
            max_displacement_rate=0.000005/2,
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

        n_neigh = np.full((4,), 3)
        nlist = np.array([
            [1, 2, 3],
            [0, 2, 3],
            [0, 1, 3],
            [0, 1, 2]
            ])

        actual = initial_crack(self.coords, nlist, n_neigh)
        expected = [
            (0, 3),
            (1, 2),
            (1, 3),
            (2, 3)
            ]

        assert expected == actual

    def test_neighbourhood(self):
        """Test with a neighbourlist defined."""
        @initial_crack_helper
        def initial_crack(icoord, jcoord):
            critical_distance = 1.0
            if np.sum((jcoord - icoord)**2) > critical_distance:
                return True
            else:
                return False

        # Create a neighbourhood matrix, ensure that particle 3 is not in the
        # neighbourhood of any other nodes
        n_neigh = np.array([2, 2, 2, 0])
        nlist = np.array([
            [1, 2],
            [0, 2],
            [0, 1],
            [0, 0]
            ])

        actual = initial_crack(self.coords, nlist, n_neigh)
        expected = [
            (1, 2)
            ]

        assert expected == actual

"""Tests for the model class."""
from .conftest import context_available
from ..model import (Model, DimensionalityError, FamilyError,
                     initial_crack_helper, InvalidIntegrator, DamageModelError)
from pyopencl import mem_flags as mf
import pyopencl as cl
from ..integrators import Euler, EulerCL, EulerCromerCL
import meshio
import numpy as np
import pytest


@pytest.fixture(
    scope="session",
    params=[Euler, pytest.param(EulerCL, marks=context_available)])
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
    """Create a basic 2D model object using a cython integrator."""
    mesh_file = data_path / "example_mesh.vtk"
    euler = Euler(dt=1e-3)
    model = Model(mesh_file, integrator=euler, horizon=0.1,
                  critical_stretch=0.05,
                  bond_stiffness=18.0 * 0.05 / (np.pi * 0.1**4),
                  is_displacement_boundary=simple_displacement_boundary)
    return model, euler


@pytest.fixture()
def basic_model_2d_cl(data_path, simple_displacement_boundary):
    """Create a basic 2D model object using an OpenCL integrator."""
    mesh_file = data_path / "example_mesh.vtk"
    euler = EulerCL(dt=1e-3)
    model = Model(mesh_file, integrator=euler, horizon=0.1,
                  critical_stretch=0.05,
                  bond_stiffness=18.0 * 0.05 / (np.pi * 0.1**4),
                  is_displacement_boundary=simple_displacement_boundary)
    return model, euler


@pytest.fixture(
    scope="session",
    params=[Euler, pytest.param(EulerCL, marks=context_available)])
def basic_models_3d(data_path, simple_displacement_boundary, request):
    """Create a basic 2D model object using a cython integrator."""
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
    return model, euler


@pytest.fixture()
def basic_model_3d_cl(data_path, simple_displacement_boundary):
    """Create a basic 3D model object using an OpenCL integrator."""
    mesh_file = data_path / "example_mesh_3d.vtk"
    euler = EulerCL(dt=1e-3)
    model = Model(mesh_file, integrator=euler, horizon=0.1,
                  critical_stretch=0.05,
                  bond_stiffness=18.0 * 0.05 / (np.pi * 0.1**4),
                  dimensions=3,
                  is_displacement_boundary=simple_displacement_boundary)
    return model, euler


@pytest.fixture(
    scope="session",
    params=[Euler, pytest.param(EulerCL, marks=context_available)])
def basic_models_transfinite(data_path, request, simple_displacement_boundary):
    """Create a basic 2D transfinite model object."""
    mesh_file = data_path / "example_mesh_transfinite.msh"
    euler = request.param(dt=1e-3)
    model = Model(mesh_file, integrator=euler, horizon=0.1,
                  critical_stretch=0.05,
                  bond_stiffness=18.0 * 0.05 / (np.pi * 0.1**4),
                  is_displacement_boundary=simple_displacement_boundary,
                  transfinite=1,
                  volume_total=1.0)
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


def test_read_transfinite(basic_models_transfinite):
    """Test reading a mesh with a transfinite model."""
    model = basic_models_transfinite

    assert model.coords.shape == (2116, 3)
    assert model.nnodes == 2116
    assert np.allclose(model.coords[69], np.array(
        [0, 0.4888888888888889, 0]))


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


class TestVolume:
    """Tests for the calculation of particle volumes."""

    def test_volume_2d(self, basic_models_2d, data_path):
        """Test volume calculation."""
        expected_volume = np.load(data_path/"expected_volume.npy")
        assert np.allclose(basic_models_2d.volume, expected_volume)

    def test_volume_3d(self, basic_models_3d, data_path):
        """Test volume calculation."""
        expected_volume = np.load(data_path/"expected_volume_3d.npy")
        assert np.allclose(basic_models_3d.volume, expected_volume)

    def test_volume_transfinite(self, basic_models_transfinite, data_path):
        """Test volume calculation."""
        expected_volume = np.ones(2116) / 2116.
        assert np.allclose(basic_models_transfinite.volume, expected_volume)

    @pytest.mark.parametrize(
        "integrator", [Euler, pytest.param(EulerCL, marks=context_available)])
    def test_no_volume_total_transfinite(self, data_path, integrator):
        """Test exceptn when volume_total not provided in transfinite mode."""
        with pytest.raises(TypeError) as exception:
            integrator = integrator(1)
            mesh_file = data_path / "example_mesh_3d.vtk"
            Model(mesh_file, integrator, horizon=0.1, critical_stretch=0.05,
                  bond_stiffness=18.0 * 0.05 / (np.pi * 0.0001**4),
                  dimensions=3, transfinite=1)
            assert (str("In transfinite mode, a total mesh volume")
                    in exception.value)

    @pytest.mark.parametrize(
        "integrator", [Euler, pytest.param(EulerCL, marks=context_available)])
    def test_volume_read(self, data_path, integrator):
        """Test reading density from file behaves as expected."""
        mesh_file = data_path / "example_mesh_3d.vtk"
        expected_volume = np.load(data_path/"expected_volume_3d.npy")

        with pytest.warns(UserWarning, match='Reading volume from argument'):
            integrator = integrator(1)
            model = Model(mesh_file, integrator=integrator, horizon=0.1,
                          critical_stretch=1.0,
                          bond_stiffness=1.0,
                          volume=expected_volume)
            assert np.allclose(model.volume, expected_volume)


def test_bond_stiffness_2d(basic_models_2d):
    """Test bond stiffness calculation."""
    assert np.isclose(basic_models_2d.bond_stiffness, 2864.7889756)


class TestConnectivity:
    """Test the initial neighbour list."""

    def test_basic_connectivity(self, basic_model_2d, data_path):
        """Test connectivity calculation with no initial crack."""
        model, integrator = basic_model_2d
        actual_nlist, actual_n_neigh = model.initial_connectivity
        npz_file = np.load(
            data_path/"expected_connectivity_basic.npz"
            )
        expected_nlist = npz_file["nlist"]
        expected_n_neigh = npz_file["n_neigh"]

        assert np.all(expected_nlist == actual_nlist)
        assert np.all(expected_n_neigh == actual_n_neigh)

    def test_basic_connectivity_cl(self, basic_model_2d_cl, data_path):
        """Test connectivity calculation with no initial crack."""
        model, integrator = basic_model_2d_cl
        actual_nlist, actual_n_neigh = model.initial_connectivity
        npz_file = np.load(
            data_path/"expected_connectivity_basic_cl.npz"
            )
        expected_nlist = npz_file["nlist"]
        expected_n_neigh = npz_file["n_neigh"]

        assert np.all(expected_nlist == actual_nlist)
        assert np.all(expected_n_neigh == actual_n_neigh)

    def test_connectivity(self, cython_model, data_path):
        """Test connectivity calculation with initial crack."""
        actual_nlist, actual_n_neigh = cython_model.initial_connectivity
        npz_file = np.load(
            data_path/"expected_connectivity_crack.npz"
            )
        expected_nlist = npz_file["nlist"]
        expected_n_neigh = npz_file["n_neigh"]

        assert np.all(expected_nlist == actual_nlist)
        assert np.all(expected_n_neigh == actual_n_neigh)

    def test_connectivity_cl(self, cl_model, data_path):
        """Test connectivity calculation with initial crack."""
        actual_nlist, actual_n_neigh = cl_model.initial_connectivity
        npz_file = np.load(
            data_path/"expected_connectivity_crack_cl.npz"
            )
        expected_nlist = npz_file["nlist"]
        expected_n_neigh = npz_file["n_neigh"]

        assert np.all(expected_nlist == actual_nlist)
        assert np.all(expected_n_neigh == actual_n_neigh)


def test_initial_damage_2d(basic_model_2d):
    """Ensure initial damage is zero."""
    model, integrator = basic_model_2d
    connectivity = model.initial_connectivity
    damage = integrator._damage(connectivity[1])

    assert np.all(damage == 0)


def test_initial_damage_3d(basic_model_3d):
    """Ensure initial damage is zero."""
    model, integrator = basic_model_3d
    connectivity = model.initial_connectivity
    damage = integrator._damage(connectivity[1])

    assert np.all(damage == 0)


@context_available
def test_initial_damage_2d_cl(basic_model_2d_cl):
    """Ensure initial damage is zero."""
    model, integrator = basic_model_2d_cl
    context = integrator.context
    queue = integrator.queue
    nlist, n_neigh = model.initial_connectivity
    local_mem = cl.LocalMemory(
        np.dtype(np.float64).itemsize * model.max_neighbours)
    family_d = cl.Buffer(
        context, mf.READ_ONLY | mf.COPY_HOST_PTR,
        hostbuf=model.family)
    nlist_d = cl.Buffer(
        context, mf.READ_WRITE | mf.COPY_HOST_PTR,
        hostbuf=nlist)
    damage = np.empty(n_neigh.shape, dtype=np.float64)
    damage_d = cl.Buffer(context, mf.WRITE_ONLY, damage.nbytes)
    n_neigh_d = cl.Buffer(context, mf.WRITE_ONLY, n_neigh.nbytes)
    integrator._damage(
        nlist_d, family_d, n_neigh_d, damage_d, local_mem)
    cl.enqueue_copy(queue, damage, damage_d)

    assert np.all(damage == 0)


@context_available
def test_initial_damage_3d_cl(basic_model_3d_cl):
    """Ensure initial damage is zero."""
    model, integrator = basic_model_3d_cl
    context = integrator.context
    queue = integrator.queue
    nlist, n_neigh = model.initial_connectivity
    local_mem = cl.LocalMemory(
        np.dtype(np.float64).itemsize * model.max_neighbours)
    family_d = cl.Buffer(
        context, mf.READ_ONLY | mf.COPY_HOST_PTR,
        hostbuf=model.family)
    nlist_d = cl.Buffer(
        context, mf.READ_WRITE | mf.COPY_HOST_PTR,
        hostbuf=nlist)
    damage = np.empty(n_neigh.shape, dtype=np.float64)
    damage_d = cl.Buffer(context, mf.WRITE_ONLY, damage.nbytes)
    n_neigh_d = cl.Buffer(context, mf.WRITE_ONLY, n_neigh.nbytes)
    integrator._damage(
        nlist_d, family_d, n_neigh_d, damage_d, local_mem)
    cl.enqueue_copy(queue, damage, damage_d)

    assert np.all(damage == 0)


@pytest.mark.parametrize(
    "integrator", [Euler, pytest.param(EulerCL, marks=context_available)])
def test_family_error(data_path, integrator):
    """Test raising of exception when a node has no neighbours."""
    with pytest.raises(FamilyError):
        integrator = integrator(1)
        mesh_file = data_path / "example_mesh_3d.vtk"
        Model(mesh_file, integrator, horizon=0.0001, critical_stretch=0.05,
              bond_stiffness=18.0 * 0.05 / (np.pi * 0.0001**4),
              dimensions=3)


@pytest.mark.parametrize(
    "integrator", [Euler, pytest.param(EulerCL, marks=context_available)])
def test_take_a_while_warning(data_path, integrator):
    """Test warning when nnodes is large."""
    with pytest.warns(UserWarning, match='this may take a while'):
        integrator = integrator(1)
        mesh_file = data_path / "example_mesh_large.msh"
        Model(mesh_file, integrator=integrator, horizon=0.1,
              critical_stretch=1.0,
              bond_stiffness=1.0)


class TestBondTypes:
    """Test _set_bond_types."""

    @pytest.mark.parametrize(
        "integrator", [Euler, pytest.param(EulerCL, marks=context_available)])
    def test_invalid_bond_type_function(
            self, data_path, request, integrator):
        """Test exception for invalid is_bond_type function."""
        mesh_file = data_path / "example_mesh.vtk"
        integrator = integrator(dt=1e-3)
        invalid_bond_type_function = 1
        with pytest.raises(TypeError) as exception:
            Model(
                mesh_file, integrator=integrator, horizon=0.1,
                critical_stretch=[[1.0], [2.0]],
                bond_stiffness=[[1.0], [2.0]],
                is_bond_type=invalid_bond_type_function)
            assert(str("is_bond_type must be a *function*.")
                   in exception.value)

    @pytest.mark.parametrize(
        "integrator", [Euler, pytest.param(EulerCL, marks=context_available)])
    def test_invalid_bond_type_function2(
            self, data_path, request, integrator):
        """Test expection for invalid is_bond_type function."""
        mesh_file = data_path / "example_mesh.vtk"
        integrator = integrator(dt=1e-3)

        def invalid_bond_type_function(x, y):
            return 1.0
        with pytest.raises(TypeError) as exception:
            Model(
                mesh_file, integrator=integrator, horizon=0.1,
                critical_stretch=[[1.0], [2.0]],
                bond_stiffness=[[1.0], [2.0]],
                is_bond_type=invalid_bond_type_function)
            assert(str(
                "is_bond_type must be a function that returns an *int*")
                   in exception.value)

    @pytest.mark.parametrize(
        "integrator", [Euler, pytest.param(EulerCL, marks=context_available)])
    def test_invalid_bond_type_function3(
            self, data_path, request, integrator):
        """Test expection for invalid is_bond_type function."""
        mesh_file = data_path / "example_mesh.vtk"
        integrator = integrator(dt=1e-3)

        def invalid_bond_type_function(x, y):
            return -1
        with pytest.raises(ValueError) as exception:
            Model(
                mesh_file, integrator=integrator, horizon=0.1,
                critical_stretch=[[1.0], [2.0]],
                bond_stiffness=[[1.0], [2.0]],
                is_bond_type=invalid_bond_type_function)
            assert(str(
                "is_bond_type must be a function that returns a *positive*")
                   in exception.value)

    @pytest.mark.parametrize(
        "integrator", [Euler, pytest.param(EulerCL, marks=context_available)])
    def test_invalid_bond_type_function4(
            self, data_path, request, integrator):
        """Test expection for invalid is_bond_type function."""
        mesh_file = data_path / "example_mesh.vtk"
        integrator = integrator(dt=1e-3)

        def invalid_bond_type_function(x, y):
            if x[0] == 0.0:
                return 0
            elif x[0] == 1.0:
                return 1
            else:
                return 2

        with pytest.raises(ValueError) as exception:
            Model(
                mesh_file, integrator=integrator, horizon=0.1,
                critical_stretch=[[0.05], [0.05]],
                bond_stiffness=[[1.0], [2.0]],
                is_bond_type=invalid_bond_type_function)
            assert(str(
                "nbond_types = 2, got is_bond_type = 2 for")
                   in exception.value)

    @context_available
    def test_valid_bond_type_function1(self, data_path, request):
        """Test expection for valid is_bond_type function."""
        mesh_file = data_path / "example_mesh.vtk"
        integrator = EulerCL(dt=1e-3)

        def invalid_bond_type_function(x, y):
            if x[0] == 0.0:
                return 0
            else:
                return 1

        model = Model(
                mesh_file, integrator=integrator, horizon=0.1,
                critical_stretch=[[0.05], [0.05]],
                bond_stiffness=[[1.0], [2.0]],
                is_bond_type=invalid_bond_type_function)
        actual_bond_types = model.bond_types
        expected_bond_types = np.load(data_path/"expected_bond_types.npy")
        assert np.all(expected_bond_types == actual_bond_types)

    @context_available
    def test_valid_bond_type_function2(self, data_path, request):
        """Test bond_types when nbond_types == 1 function."""
        mesh_file = data_path / "example_mesh.vtk"
        integrator = EulerCL(dt=1e-3)

        model = Model(
                mesh_file, integrator=integrator, horizon=0.1,
                critical_stretch=[0.05, 0.10],
                bond_stiffness=[1.0, 0.5])
        actual_bond_types = model.bond_types
        expected_bond_types = np.zeros((model.nnodes, model.max_neighbours),
                                       dtype=np.intc)
        assert np.all(expected_bond_types == actual_bond_types)

    @pytest.mark.parametrize(
        "integrator", [Euler, pytest.param(EulerCL, marks=context_available)])
    def test_valid_bond_type_function3(
            self, data_path, request, integrator):
        """Test bond_types when nbond_types == 1 and nregimes == 1."""
        mesh_file = data_path / "example_mesh.vtk"
        integrator = integrator(dt=1e-3)

        model = Model(
                mesh_file, integrator=integrator, horizon=0.1,
                critical_stretch=0.05,
                bond_stiffness=1.0)
        actual_bond_types = model.bond_types
        assert actual_bond_types is None

    def test_bond_type_support(self, data_path):
        """Test _set_bond_types support for the Euler integrator."""
        mesh_file = data_path / "example_mesh.vtk"
        integrator = Euler(dt=1e-3)

        def bond_type_function(x, y):
            if x[0] == 0.0:
                return 0
            else:
                return 1
        with pytest.raises(ValueError) as exception:
            Model(mesh_file, integrator=integrator, horizon=0.1,
                  critical_stretch=[[0.05], [0.05]],
                  bond_stiffness=[[1.0], [2.0]],
                  is_bond_type=bond_type_function)

            assert(str("bond_types are not supported by this")
                   in exception.value)

    @context_available
    def test_change_nbond_types(self, data_path):
        """Test exception when nbond_types is changed between simulations."""
        mesh_file = data_path / "example_mesh.vtk"
        integrator = EulerCromerCL(dt=1, damping=1)

        def bond_type_function(x, y):
            if x[0] == 0.0:
                return 0
            else:
                return 1

        def density_function(x):
            return 1.0

        model = Model(mesh_file, integrator=integrator, horizon=0.1,
                      critical_stretch=[[0.05], [0.05]],
                      bond_stiffness=[[1.0], [2.0]],
                      is_bond_type=bond_type_function,
                      is_density=density_function)
        with pytest.raises(ValueError) as exception:
            model.simulate(steps=2,
                           bond_stiffness=[[1.0], [2.0], [3.0]],
                           critical_stretch=[[0.05], [0.05], [0.05]])
            assert(str("Number of bond types has unexpectedly changed")
                   in exception.value)


class TestDensities:
    """Test _set_densities."""

    @pytest.mark.parametrize(
        "integrator", [Euler, pytest.param(EulerCL, marks=context_available)])
    def test_invalid_density_type_function(
            self, data_path, request, integrator):
        """Test exception for invalid is_density function."""
        mesh_file = data_path / "example_mesh.vtk"
        integrator = integrator(dt=1e-3)
        invalid_density_function = 1
        with pytest.raises(TypeError) as exception:
            Model(
                mesh_file, integrator=integrator, horizon=0.1,
                critical_stretch=1.0,
                bond_stiffness=1.0,
                is_density=invalid_density_function)
            assert(str("is_density must be a *function*.")
                   in exception.value)

    @pytest.mark.parametrize(
        "integrator", [Euler, pytest.param(EulerCL, marks=context_available)])
    def test_invalid_density_function2(
            self, data_path, request, integrator):
        """Test exception for invalid is_density function."""
        mesh_file = data_path / "example_mesh.vtk"
        integrator = integrator(dt=1e-3)

        def invalid_density_function(x):
            return 1
        with pytest.raises(TypeError) as exception:
            Model(
                mesh_file, integrator=integrator, horizon=0.1,
                critical_stretch=1.0,
                bond_stiffness=1.0,
                is_density=invalid_density_function)
            assert(str(
                "is_density must be a function that returns a *float*")
                   in exception.value)

    @pytest.mark.parametrize(
        "integrator", [Euler, pytest.param(EulerCL, marks=context_available)])
    def test_invalid_density(
            self, data_path, request, integrator):
        """Test invalid density."""
        mesh_file = data_path / "example_mesh.vtk"
        integrator = integrator(dt=1e-3)
        density = 1.0
        with pytest.raises(TypeError) as exception:
            Model(
                mesh_file, integrator=integrator, horizon=0.1,
                critical_stretch=[[0.05], [0.05]],
                bond_stiffness=[[1.0], [2.0]],
                density=density)
            assert(str("density type is wrong, and must be an array of")
                   in exception.value)

    @pytest.mark.parametrize(
        "integrator", [Euler, pytest.param(EulerCL, marks=context_available)])
    def test_invalid_density2(
            self, data_path, request, integrator):
        """Test invalid density shape."""
        mesh_file = data_path / "example_mesh.vtk"
        integrator = integrator(dt=1e-3)
        density = np.ones(2)
        with pytest.raises(ValueError) as exception:
            Model(
                mesh_file, integrator=integrator, horizon=0.1,
                critical_stretch=[[0.05], [0.05]],
                bond_stiffness=[[1.0], [2.0]],
                density=density)
            assert(str("densty shape is wrong, and must be")
                   in exception.value)

    def test_density_support_euler(self, data_path):
        """Test densities support for the Euler integrator."""
        mesh_file = data_path / "example_mesh.vtk"
        integrator = Euler(dt=1e-3)

        def density_function(x):
            if x[0] == 0.0:
                return 1.0
            else:
                return 2.0
        with pytest.raises(ValueError) as exception:
            Model(mesh_file, integrator=integrator, horizon=0.1,
                  critical_stretch=1.0,
                  bond_stiffness=1.0,
                  is_density=density_function)
            assert(str("densities are not supported by this ")
                   in exception.value)

    @context_available
    def test_density_support_euler_cl(self, data_path):
        """Test densities support for the EulerCL integrator."""
        mesh_file = data_path / "example_mesh.vtk"
        integrator = EulerCL(dt=1e-3)

        def density_function(x):
            if x[0] == 0.0:
                return 1.0
            else:
                return 2.0
        with pytest.raises(ValueError) as exception:
            Model(mesh_file, integrator=integrator, horizon=0.1,
                  critical_stretch=1.0,
                  bond_stiffness=1.0,
                  is_density=density_function)
            assert(str("densities are not supported by this ")
                   in exception.value)

    @context_available
    def test_density_support_euler_cromer_cl(self, data_path):
        """Test densities support for the EulerCromerCL integrator."""
        mesh_file = data_path / "example_mesh.vtk"
        integrator = EulerCromerCL(dt=1e-3, damping=1.0)
        expected_densities = np.load(
            data_path / "expected_densities.npy")

        def density_function(x):
            if x[0] == 0.0:
                return 1.0
            else:
                return 2.0
        model = Model(mesh_file, integrator=integrator, horizon=0.1,
                      critical_stretch=1.0,
                      bond_stiffness=1.0,
                      is_density=density_function)
        actual_densities = model.densities

        assert np.all(actual_densities == expected_densities)

    @context_available
    def test_density_read_cl(self, data_path):
        """Test reading density from file behaves as expected."""
        mesh_file = data_path / "example_mesh.vtk"
        integrator = EulerCromerCL(dt=1e-3, damping=1.0)
        expected_densities = np.load(
            data_path / "expected_densities.npy")

        density = expected_densities[:, 0]

        with pytest.warns(UserWarning, match='Reading density from argument'):
            model = Model(mesh_file, integrator=integrator, horizon=0.1,
                          critical_stretch=1.0,
                          bond_stiffness=1.0,
                          density=density)
            actual_densities = model.densities

            assert np.all(actual_densities == expected_densities)

    @context_available
    def test_density_support_euler_cromer_cl2(self, data_path):
        """Test exception no densities supplied EulerCromerCL integrator."""
        mesh_file = data_path / "example_mesh.vtk"
        integrator = EulerCromerCL(dt=1e-3, damping=1.0)

        with pytest.raises(ValueError) as exception:
            Model(mesh_file, integrator=integrator, horizon=0.1,
                  critical_stretch=1.0,
                  bond_stiffness=1.0)
            assert (str("densities must be supplied when using EulerCromerCL")
                    in exception.value)


class TestStiffnessCorrections:
    """Test _set_stiffness_corrections."""

    def test_value_stiffness_correction(self, data_path):
        """Test exception when precise stiffness correction value is wrong."""
        with pytest.raises(ValueError) as exception:
            integrator = Euler(1)
            mesh_file = data_path / "example_mesh_3d.vtk"
            Model(mesh_file, integrator, horizon=0.1, critical_stretch=0.05,
                  bond_stiffness=18.0 * 0.05 / (np.pi * 0.0001**4),
                  dimensions=3, precise_stiffness_correction=2)
            assert(str("precise_stiffness_correction value is wrong")
                   in exception.value)

    def test_inprecise_stiffness_correction_2d(
            self, basic_model_2d, data_path):
        """Test stiffness corrections using average nodal volumes."""
        model, integrator = basic_model_2d
        actual_stiffness_corrections = model._set_stiffness_corrections(
            precise_stiffness_correction=0,
            corrections=np.ones(
                (model.nnodes, model.max_neighbours), dtype=np.float64))
        expected_stiffness_corrections = np.load(
            data_path / "expected_stiffness_corrections_2d.npy")
        assert np.allclose(
            expected_stiffness_corrections,
            actual_stiffness_corrections)

    def test_precise_stiffness_correction_2d(
            self, basic_model_2d, data_path):
        """Test stiffness corrections using precise nodal volumes."""
        model, integrator = basic_model_2d
        actual_stiffness_corrections = model._set_stiffness_corrections(
            precise_stiffness_correction=1,
            corrections=np.ones(
                (model.nnodes, model.max_neighbours), dtype=np.float64))
        expected_stiffness_corrections = np.load(
            data_path / "expected_stiffness_corrections_2d_precise.npy")
        assert np.allclose(
            expected_stiffness_corrections,
            actual_stiffness_corrections)

    def test_inprecise_stiffness_correction_3d(
            self, basic_model_3d, data_path):
        """Test stiffness corrections using average nodal volumes."""
        model, integrator = basic_model_3d
        actual_stiffness_corrections = model._set_stiffness_corrections(
            precise_stiffness_correction=0,
            corrections=np.ones(
                (model.nnodes, model.max_neighbours), dtype=np.float64))
        expected_stiffness_corrections = np.load(
            data_path / "expected_stiffness_corrections_3d.npy")
        assert np.allclose(
            expected_stiffness_corrections,
            actual_stiffness_corrections)

    @context_available
    def test_inprecise_stiffness_correction_2d_cl(
            self, basic_model_2d_cl, data_path):
        """Test stiffness corrections using average nodal volumes."""
        model, integrator = basic_model_2d_cl
        actual_stiffness_corrections = model._set_stiffness_corrections(
            precise_stiffness_correction=0,
            corrections=np.ones(
                (model.nnodes, model.max_neighbours), dtype=np.float64))
        expected_stiffness_corrections = np.load(
            data_path / "expected_stiffness_corrections_2d_cl.npy")
        assert np.allclose(
            expected_stiffness_corrections,
            actual_stiffness_corrections)

    @context_available
    def test_precise_stiffness_correction_2d_cl(
            self, basic_model_2d_cl, data_path):
        """Test stiffness corrections using precise nodal volumes."""
        model, integrator = basic_model_2d_cl
        actual_stiffness_corrections = model._set_stiffness_corrections(
            precise_stiffness_correction=1,
            corrections=np.ones(
                (model.nnodes, model.max_neighbours), dtype=np.float64))
        expected_stiffness_corrections = np.load(
            data_path / "expected_stiffness_corrections_2d_precise_cl.npy")
        assert np.allclose(
            expected_stiffness_corrections,
            actual_stiffness_corrections)

    @context_available
    def test_inprecise_stiffness_correction_3d_cl(
            self, basic_model_3d_cl, data_path):
        """Test stiffness corrections using average nodal volumes."""
        model, integrator = basic_model_3d_cl
        actual_stiffness_corrections = model._set_stiffness_corrections(
            precise_stiffness_correction=0,
            corrections=np.ones(
                (model.nnodes, model.max_neighbours), dtype=np.float64))
        expected_stiffness_corrections = np.load(
            data_path / "expected_stiffness_corrections_3d_cl.npy")
        assert np.allclose(
            expected_stiffness_corrections,
            actual_stiffness_corrections)

    @context_available
    def test_precise_stiffness_correction_3d_cl(
            self, basic_model_3d_cl, data_path):
        """Test stiffness corrections using precise nodal volumes."""
        model, integrator = basic_model_3d_cl
        actual_stiffness_corrections = model._set_stiffness_corrections(
            precise_stiffness_correction=1,
            corrections=np.ones(
                (model.nnodes, model.max_neighbours), dtype=np.float64))
        expected_stiffness_corrections = np.load(
            data_path / "expected_stiffness_corrections_3d_precise_cl.npy")
        assert np.allclose(
            expected_stiffness_corrections,
            actual_stiffness_corrections)

    def test_none_stiffness_corrections(self, data_path):
        """Test no stiffness corrections case."""
        integrator = Euler(1)
        mesh_file = data_path / "example_mesh_3d.vtk"
        model = Model(mesh_file, integrator, horizon=0.1,
                      critical_stretch=0.05,
                      bond_stiffness=18.0 * 0.05 / (np.pi * 0.0001**4),
                      dimensions=3)
        actual_stiffness_corrections = model.stiffness_corrections
        assert actual_stiffness_corrections is None


class TestDamageModel:
    """Test _set_damage_model."""

    def test_plus_cs_list(self, basic_models_2d):
        """Test for damage model parameters as lists."""
        bond_stiffness = [[1.0, -1.0, -0.5], [1.0, 0.0, 0.0]]
        critical_stretch = [[1.0, 1.5, 2.5], [1.0, 1000.0, 1001.0]]

        bond_stiffness_expected = np.array(bond_stiffness, dtype=np.float64)
        critical_stretch_expected = np.array(
            critical_stretch, dtype=np.float64)
        plus_cs_expected = np.array([[0.0, 2.0, 1.25],
                                     [0.0, 1.0, 1.0]])
        nregimes_expected = np.intc(3)
        nbond_types_expected = np.intc(2)

        model = basic_models_2d
        (bond_stiffness_actual,
         critical_stretch_actual,
         plus_cs_actual,
         nbond_types_actual,
         nregimes_actual) = model._set_damage_model(bond_stiffness,
                                                    critical_stretch)

        assert np.all(bond_stiffness_actual == bond_stiffness_expected)
        assert np.all(critical_stretch_actual == critical_stretch_expected)
        assert nregimes_actual == nregimes_expected
        assert nbond_types_actual == nbond_types_expected
        assert np.all(plus_cs_actual == plus_cs_expected)

    def test_plus_cs_array(self, basic_models_2d):
        """Test for damage model parameters as arrays."""
        bond_stiffness = np.array([[1.0, -1.0, -0.5], [1.0, 0.0, 0.0]])
        critical_stretch = np.array([[1.0, 1.5, 2.5], [1.0, 1000.0, 1001.0]])

        bond_stiffness_expected = np.array(bond_stiffness, dtype=np.float64)
        critical_stretch_expected = np.array(
            critical_stretch, dtype=np.float64)
        plus_cs_expected = np.array([[0.0, 2.0, 1.25],
                                     [0.0, 1.0, 1.0]], dtype=np.float64)
        nregimes_expected = np.intc(3)
        nbond_types_expected = np.intc(2)

        model = basic_models_2d
        (bond_stiffness_actual,
         critical_stretch_actual,
         plus_cs_actual,
         nbond_types_actual,
         nregimes_actual) = model._set_damage_model(bond_stiffness,
                                                    critical_stretch)

        assert np.all(bond_stiffness_actual == bond_stiffness_expected)
        assert np.all(critical_stretch_actual == critical_stretch_expected)
        assert nregimes_actual == nregimes_expected
        assert nbond_types_actual == nbond_types_expected
        assert np.all(plus_cs_actual == plus_cs_expected)

    def test_plus_cs_float_array(self, basic_models_2d):
        """Test for damage model parameters as float array."""
        bond_stiffness = np.array(1.0)
        critical_stretch = np.array(1.0)

        bond_stiffness_expected = np.array(bond_stiffness, dtype=np.float64)
        critical_stretch_expected = np.array(
            critical_stretch, dtype=np.float64)
        plus_cs_expected = None
        nregimes_expected = np.intc(1)
        nbond_types_expected = np.intc(1)

        model = basic_models_2d
        (bond_stiffness_actual,
         critical_stretch_actual,
         plus_cs_actual,
         nbond_types_actual,
         nregimes_actual) = model._set_damage_model(bond_stiffness,
                                                    critical_stretch)

        assert np.all(bond_stiffness_actual == bond_stiffness_expected)
        assert np.all(critical_stretch_actual == critical_stretch_expected)
        assert nregimes_actual == nregimes_expected
        assert nbond_types_actual == nbond_types_expected
        assert np.all(plus_cs_actual == plus_cs_expected)

    def test_plus_cs_float_list(self, basic_models_2d):
        """Test for damage model parameters as float array."""
        bond_stiffness = [1.0]
        critical_stretch = [1.0]

        bond_stiffness_expected = np.array(bond_stiffness, dtype=np.float64)
        critical_stretch_expected = np.array(
            critical_stretch, dtype=np.float64)
        plus_cs_expected = None
        nregimes_expected = np.intc(1)
        nbond_types_expected = np.intc(1)

        model = basic_models_2d
        (bond_stiffness_actual,
         critical_stretch_actual,
         plus_cs_actual,
         nbond_types_actual,
         nregimes_actual) = model._set_damage_model(bond_stiffness,
                                                    critical_stretch)

        assert np.all(bond_stiffness_actual == bond_stiffness_expected)
        assert np.all(critical_stretch_actual == critical_stretch_expected)
        assert nregimes_actual == nregimes_expected
        assert nbond_types_actual == nbond_types_expected
        assert np.all(plus_cs_actual == plus_cs_expected)

    def test_plus_cs_one_bond(self, basic_models_2d):
        """Test for damage model parameters as float array."""
        bond_stiffness = np.array([1.0, -1.0, -0.5])
        critical_stretch = np.array([1.0, 2.0, 3.0])

        bond_stiffness_expected = np.array(bond_stiffness, dtype=np.float64)
        critical_stretch_expected = np.array(
            critical_stretch, dtype=np.float64)
        plus_cs_expected = np.array([0.0, 2.0, 1.0], dtype=np.float64)
        nregimes_expected = np.intc(3)
        nbond_types_expected = np.intc(1)

        model = basic_models_2d
        (bond_stiffness_actual,
         critical_stretch_actual,
         plus_cs_actual,
         nbond_types_actual,
         nregimes_actual) = model._set_damage_model(bond_stiffness,
                                                    critical_stretch)

        assert np.all(bond_stiffness_actual == bond_stiffness_expected)
        assert np.all(critical_stretch_actual == critical_stretch_expected)
        assert nregimes_actual == nregimes_expected
        assert nbond_types_actual == nbond_types_expected
        assert np.all(plus_cs_actual == plus_cs_expected)

    def test_plus_cs_one_regime(self, basic_models_2d):
        """Test for damage model parameters as float array."""
        bond_stiffness = np.array([[1.0], [2.0], [1.0]])
        critical_stretch = np.array([[1.0], [1.0], [1.0]])

        bond_stiffness_expected = np.array(bond_stiffness, dtype=np.float64)
        critical_stretch_expected = np.array(
            critical_stretch, dtype=np.float64)
        plus_cs_expected = np.array([[0.0], [0.0], [0.0]], dtype=np.float64)
        nregimes_expected = np.intc(1)
        nbond_types_expected = np.intc(3)

        model = basic_models_2d
        (bond_stiffness_actual,
         critical_stretch_actual,
         plus_cs_actual,
         nbond_types_actual,
         nregimes_actual) = model._set_damage_model(bond_stiffness,
                                                    critical_stretch)

        assert np.all(bond_stiffness_actual == bond_stiffness_expected)
        assert np.all(critical_stretch_actual == critical_stretch_expected)
        assert nregimes_actual == nregimes_expected
        assert nbond_types_actual == nbond_types_expected
        assert np.all(plus_cs_actual == plus_cs_expected)

    def test_plus_cs_negative_array(self, basic_models_2d):
        """Test for damage model parameters with negative critical stretch."""
        bond_stiffness = np.array([[1.0, 1.0, -0.5], [1.0, 0.0, 0.0]])
        critical_stretch = np.array([[-1.0, 1.0, 2.5], [1.0, 1000.0, 1001.0]])
        model = basic_models_2d
        with pytest.raises(ValueError) as exception:
            model._set_damage_model(bond_stiffness, critical_stretch)
            assert(str("critical_stretch values must not be < 0")
                   in exception.values)

    def test_plus_cs_unordered_array(self, basic_models_2d):
        """Test for damage model parameters with negative critical stretch."""
        bond_stiffness = np.array([[1.0, 1.0, -0.5], [1.0, 0.0, 0.0]])
        critical_stretch = np.array([[1.0, 0.9, 2.5], [1.0, 1000.0, 1001.0]])
        model = basic_models_2d
        with pytest.raises(DamageModelError) as exception:
            model._set_damage_model(bond_stiffness, critical_stretch)
            assert(str("must be in ascending order")
                   in exception.values)

    def test_plus_cs_unordered_array2(self, basic_models_2d):
        """Test for damage model parameters with negative critical stretch."""
        bond_stiffness = np.array([1.0, 1.0, -0.5])
        critical_stretch = np.array([1.0, 0.9, 2.5])
        model = basic_models_2d
        with pytest.raises(DamageModelError) as exception:
            model._set_damage_model(bond_stiffness, critical_stretch)
            assert(str("must be in ascending order")
                   in exception.values)

    def test_plus_cs_negative_float(self, basic_models_2d):
        """Test for damage model parameters with negative critical stretch."""
        bond_stiffness = 1.0
        critical_stretch = -1.0
        model = basic_models_2d
        with pytest.raises(ValueError) as exception:
            model._set_damage_model(bond_stiffness, critical_stretch)
            assert(
                str("critical_stretch values must not be < 0")
                in exception.value)

    def test_plus_cs_different_type(self, basic_models_2d):
        """Test for different types."""
        bond_stiffness = 1.0
        critical_stretch = None
        model = basic_models_2d
        with pytest.raises(TypeError) as exception:
            model._set_damage_model(bond_stiffness, critical_stretch)
            assert(
                str("bond_stiffness must be the same type") in exception.value)

    def test_plus_cs_type(self, basic_models_2d):
        """Test for incorrect type."""
        bond_stiffness = None
        critical_stretch = None
        model = basic_models_2d
        with pytest.raises(TypeError) as exception:
            model._set_damage_model(bond_stiffness, critical_stretch)
            assert(str("Type of bond_stiffness and critical_stretch"
                       " is not supported") in exception.value)

    def test_plus_cs_shape(self, basic_models_2d):
        """Test for damage model parameters with negative critical stretch."""
        bond_stiffness = np.array([[1.0, -1.0, -0.5], [1.0, 0.0, 0.0]])
        critical_stretch = np.array([[1.0, 2.0], [1.0, 2.0]])
        model = basic_models_2d
        with pytest.raises(ValueError) as exception:
            model._set_damage_model(bond_stiffness, critical_stretch)
            assert(str("The shape of bond_stiffness "
                       "must be equal to the shape") in exception.value)


class TestBoundaryConditions:
    """Tests for the _set_boundary_conditions method."""

    def test_invalid_boundary_function(self, data_path, request):
        """Test for exception for an invalid boundary function."""
        mesh_file = data_path / "example_mesh.vtk"
        euler = Euler(dt=1e-3)
        invalid_boundary_function = [None, None, None]
        with pytest.raises(TypeError) as exception:
            Model(
                mesh_file, integrator=euler, horizon=0.1,
                critical_stretch=0.05,
                bond_stiffness=18.0 * 0.05 / (np.pi * 0.1**4),
                is_displacement_boundary=invalid_boundary_function)
            assert("is_displacement_boundary must be a *function*."
                   in exception.value)

    def test_invalid_boundary_function2(self, data_path, request):
        """Test for exception for an invalid boundary function."""
        mesh_file = data_path / "example_mesh.vtk"
        euler = Euler(dt=1e-3)

        def invalid_boundary_function():
            """Return an invalid boundary function."""
            return 1

        with pytest.raises(TypeError) as exception:
            Model(
                mesh_file, integrator=euler, horizon=0.1,
                critical_stretch=0.05,
                bond_stiffness=18.0 * 0.05 / (np.pi * 0.1**4),
                is_displacement_boundary=invalid_boundary_function)
            assert("is_displacement_boundary must be a function that returns"
                   + " a *list*." in exception.value)

    def test_invalid_boundary_function3(self, data_path, request):
        """Test for exception for an invalid boundary function."""
        mesh_file = data_path / "example_mesh.vtk"
        euler = Euler(dt=1e-3)

        def invalid_boundary_function():
            """Return an invalid boundary function."""
            return [None]

        with pytest.raises(TypeError) as exception:
            Model(
                mesh_file, integrator=euler, horizon=0.1,
                critical_stretch=0.05,
                bond_stiffness=18.0 * 0.05 / (np.pi * 0.1**4),
                is_displacement_boundary=invalid_boundary_function)
            assert("{} must return a function that returns a list"
                   + " of length *3* of floats or None" in exception.value)

    def test_no_boundary_function(self, data_path, request):
        """Ensure no boundary function works as expected."""
        mesh_file = data_path / "example_mesh.vtk"
        euler = Euler(dt=1e-3)

        def is_displacement_boundary(x):
            return [None, None, None]

        def is_force_boundary(x):
            return [None, None, None]

        model = Model(
            mesh_file, integrator=euler, horizon=0.1,
            critical_stretch=0.05,
            bond_stiffness=18.0 * 0.05 / (np.pi * 0.1**4),
            is_displacement_boundary=is_displacement_boundary,
            is_force_boundary=is_force_boundary)

        u, damage, connectivity, *_ = model.simulate(
            steps=2,
            displacement_bc_magnitudes=np.array([0.000005/2, 0.000005])
            )

        model = Model(
            mesh_file, integrator=euler, horizon=0.1,
            critical_stretch=0.05,
            bond_stiffness=18.0 * 0.05 / (np.pi * 0.1**4))

        (expected_u,
         expected_damage,
         expected_connectivity,
         *_) = model.simulate(
            steps=2,
            displacement_bc_magnitudes=np.array([0.000005/2, 0.000005])
            )

        assert np.all(u == expected_u)
        assert np.all(damage == expected_damage)
        assert np.all(connectivity[0] == expected_connectivity[0])
        assert np.all(connectivity[1] == expected_connectivity[1])

    def test_boundary_function(self, data_path, request):
        """Regression test for _set_boundary_conditions function."""
        mesh_file = data_path / "example_mesh.vtk"
        euler = Euler(dt=1e-3)

        def is_displacement_boundary(x):
            bnd = [None, None, None]
            if x[0] < 0.1:
                bnd[0] = -1
            elif x[0] > 0.9:
                bnd[0] = 1
            return bnd

        def is_force_boundary(x):
            bnd = [None, None, None]
            if x[0] < 0.1:
                bnd[0] = -1
            elif x[0] > 0.9:
                bnd[0] = 1
            return bnd

        def is_tip(x):
            bnd = [None, None, None]
            if x[0] < 0.1:
                bnd[0] = 1
                bnd[1] = 2
            else:
                bnd[0] = 'otherwise'
                bnd[1] = 2
            return bnd

        model = Model(
            mesh_file, integrator=euler, horizon=0.1,
            critical_stretch=0.05,
            bond_stiffness=18.0 * 0.05 / (np.pi * 0.1**4),
            is_displacement_boundary=is_displacement_boundary,
            is_force_boundary=is_force_boundary,
            is_tip=is_tip)

        (actual_bc_types,
         actual_bc_values,
         actual_force_bc_types,
         actual_force_bc_values,
         _,
         ntips) = model._set_boundary_conditions(
            is_displacement_boundary=is_displacement_boundary,
            is_force_boundary=is_force_boundary,
            is_tip=is_tip)

        expected_bc_types = np.load(data_path/"expected_bc_types.npy")
        expected_bc_values = np.load(data_path/"expected_bc_values.npy")
        expected_force_bc_types = np.load(
            data_path/"expected_force_bc_types.npy")
        expected_force_bc_values = np.load(
            data_path/"expected_force_bc_values.npy")

        assert ntips['model'] == 2113
        assert ntips['1'] == 228
        assert ntips['otherwise'] == 2113 - 228
        assert ntips['2'] == 2113
        assert np.allclose(actual_bc_types, expected_bc_types)
        assert np.allclose(actual_bc_values, expected_bc_values)
        assert np.allclose(actual_force_bc_types, expected_force_bc_types)
        assert np.allclose(actual_force_bc_values, expected_force_bc_values)


class TestIntegrator:
    """
    Tests for the integrator.

    Further tests of integrator are in test_integrator.py
    """

    def test_invalid_integrator(self, data_path):
        """Test passing an invalid integrator to simulate."""
        mesh_file = data_path / "example_mesh.vtk"
        invalid_integrator = None
        with pytest.raises(InvalidIntegrator):
            Model(
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

    def test_stateless(self, basic_models_2d):
        """Ensure the simulate method does not affect the state of Models."""
        model = basic_models_2d
        nlist, n_neigh = model.initial_connectivity
        expected_initial_nlist = nlist.copy()
        expected_initial_n_neigh = n_neigh.copy()
        steps = 10
        (u,
         damage,
         connectivity,
         force,
         ud,
         *_) = model.simulate(
            steps=steps,
            displacement_bc_magnitudes=1 / 2 * np.linspace(
                1, steps, steps))
        initial_nlist, initial_n_neigh = model.initial_connectivity
        (expected_u,
         expected_damage,
         expected_connectivity,
         expected_force,
         expected_ud,
         *_) = model.simulate(
            steps=steps,
            displacement_bc_magnitudes=1 / 2 * np.linspace(
                1, steps, steps))
        assert np.all(u == expected_u)
        assert np.all(damage == expected_damage)
        assert np.all(connectivity[0] == expected_connectivity[0])
        assert np.all(connectivity[1] == expected_connectivity[1])
        assert np.all(initial_nlist == expected_initial_nlist)
        assert np.all(initial_n_neigh == expected_initial_n_neigh)
        assert np.all(force == expected_force)
        assert np.all(ud == expected_ud)

    @pytest.fixture(scope="module")
    def simulate_force_test(self, data_path):
        """Create a minimal model designed for testings force calculation."""
        mesh_file = data_path/"force_test.vtk"
        integrator = Euler(dt=1)
        model = Model(mesh_file, integrator, horizon=1.01,
                      critical_stretch=0.06,
                      bond_stiffness=18.0 * 0.05 / (np.pi * 1.01**4)
                      )
        return model

    def test_force(self, simulate_force_test):
        """Ensure forces are in the correct direction using a minimal model."""
        model = simulate_force_test
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

        # Simulate the model for one time step
        u, damage, connectivity, force, *_ = model.simulate(
            steps=1, u=u)

        # Ensure force array is correct
        force_value = 0.00229417
        expected_force = np.array([
            [force_value, 0., 0.],
            [-force_value, force_value, 0.],
            [0., -force_value, 0.]
            ])
        assert np.allclose(force, expected_force)

        # Ensure force is restorative,
        #   - Node 1 pulls node 0 in the positive x direction
        #   - Node 0 pulls node 1 in the negative x direction
        assert force[0, 0] > 0
        assert force[1, 0] < 0
        #   - Node 2 pulls node 1 in the positive y direction
        #   - Node 1 pulls node 2 in the negative y direction
        assert force[1, 1] > 0
        assert force[2, 1] < 0

        # Node 0 has no component of force in the y or z dimensions
        assert np.all(force[0, 1:] == 0)
        # Node 1 has no component of force in the z dimension
        assert force[1, 2] == 0
        # Node 2 has no component of force in the x or z dimensions
        assert np.all(force[2, [0, 2]] == 0)

    def test_restart(self, basic_models_2d):
        """Ensure simulation restarting gives consistent results."""
        model = basic_models_2d
        displacement_bc_magnitudes = np.linspace(0, (0.00001 * 99 / 2), 100)

        (u,
         damage,
         connectivity,
         force,
         ud,
         *_) = model.simulate(
            steps=1,
            displacement_bc_magnitudes=displacement_bc_magnitudes
            )
        (u,
         damage,
         connectivity,
         force,
         ud,
         *_) = model.simulate(
            steps=1,
            displacement_bc_magnitudes=displacement_bc_magnitudes,
            u=u,
            connectivity=connectivity,
            first_step=2
            )

        (expected_u,
         expected_damage,
         expected_connectivity,
         expected_force,
         expected_ud,
         *_) = model.simulate(
            steps=2,
            displacement_bc_magnitudes=displacement_bc_magnitudes
            )

        assert np.all(u == expected_u)
        assert np.all(damage == expected_damage)
        assert np.all(connectivity[0] == expected_connectivity[0])
        assert np.all(connectivity[1] == expected_connectivity[1])
        assert np.all(force == expected_force)
        assert np.all(ud == expected_ud)

    def test_write(self, cython_model, tmp_path):
        """Ensure that the mesh file written by simulate is correct."""
        model = cython_model

        u, damage, connectivity, *_ = model.simulate(
            steps=1,
            displacement_bc_magnitudes=np.array([0]),
            write=1,
            write_path=tmp_path
            )

        mesh = tmp_path / "U_1.vtk"

        expected_mesh = tmp_path / "mesh.vtk"
        model.write_mesh(expected_mesh, damage, u)

        assert mesh.read_bytes() == expected_mesh.read_bytes()


class TestSimulateInitialise:
    """Tests for the _simulate_initialise function."""

    def test_data(self, cython_model):
        """Ensure data dict contains the correct sized containers."""
        model = cython_model
        steps = 20
        write = 10
        (_, _, _, _, _, _, _, _, _, _,
         actual_data, actual_nwrites, *_) = model._simulate_initialise(
            steps, 1, write, None, None, None,
            None, None, None, None, None, None)
        expected_data_model = {'step': np.zeros(2, dtype=int),
                               'displacement': np.zeros(2, dtype=np.float64),
                               'velocity': np.zeros(2, dtype=np.float64),
                               'acceleration': np.zeros(2, dtype=np.float64),
                               'force': np.zeros(2, dtype=np.float64),
                               'body_force': np.zeros(2, dtype=np.float64),
                               'damage_sum': np.zeros(2, dtype=np.float64)}
        actual_data_model = actual_data['model']

        assert actual_data_model.keys() == expected_data_model.keys()
        assert all(
            np.array_equal(actual_data_model[key], expected_data_model[key])
            for key in actual_data_model)
        assert actual_nwrites == 2

    def test_data_first_step(self, cython_model):
        """Ensure data dict contains the correct sized containers."""
        model = cython_model
        steps = 5
        first_step = 6
        write = 10
        (_, _, _, _, _, _, _, _, _, _,
         actual_data, actual_nwrites, *_) = model._simulate_initialise(
            steps, first_step, write, None, None, None,
            None, None, None, None, None, None)
        expected_data_model = {'step': np.zeros(1, dtype=int),
                               'displacement': np.zeros(1, dtype=np.float64),
                               'velocity': np.zeros(1, dtype=np.float64),
                               'acceleration': np.zeros(1, dtype=np.float64),
                               'force': np.zeros(1, dtype=np.float64),
                               'body_force': np.zeros(1, dtype=np.float64),
                               'damage_sum': np.zeros(1, dtype=np.float64)}
        actual_data_model = actual_data['model']

        assert actual_data_model.keys() == expected_data_model.keys()
        assert all(
            np.array_equal(actual_data_model[key], expected_data_model[key])
            for key in actual_data_model)
        assert actual_nwrites == 1

    def test_none_initalise(self, cython_model):
        """Ensure simulate values are initialised correctly given None."""
        model = cython_model
        nlist_expected, n_neigh_expected = model.initial_connectivity
        steps = 10
        (u, ud, udd, force, body_force, nlist, n_neigh,
         displacement_bc_magnitudes, force_bc_magnitudes, damage, data,
         nwrites, write_path) = model._simulate_initialise(
            steps, 1, None, None, None, None,
            None, None, None, None, None, None)
        nnodes_dof_zeros = np.zeros((model.nnodes, 3), dtype=np.float64)
        nnodes_zeros = np.zeros((model.nnodes), dtype=np.float64)
        steps_zeros = np.zeros(steps, dtype=np.float64)
        assert np.all(u == nnodes_dof_zeros)
        assert np.all(ud == nnodes_dof_zeros)
        assert np.all(udd == nnodes_dof_zeros)
        assert np.all(force == nnodes_dof_zeros)
        assert np.all(body_force == nnodes_dof_zeros)
        assert np.all(nlist == nlist_expected)
        assert np.all(n_neigh == n_neigh_expected)
        assert np.all(displacement_bc_magnitudes == steps_zeros)
        assert np.all(force_bc_magnitudes == steps_zeros)
        assert np.all(damage == nnodes_zeros)
        assert data == {}
        assert nwrites is None

    def test_displacement_bc_magnitudes_length(self, cython_model):
        """Test exception when displacement_bc_magnitudes length is wrong."""
        model = cython_model
        nlist_expected, n_neigh_expected = model.initial_connectivity
        steps = 10
        displacement_bc_magnitudes = np.zeros(9)
        with pytest.raises(ValueError) as exception:
            model._simulate_initialise(
                steps, 1, None, None, None, None,
                displacement_bc_magnitudes, None, None, None, None, None)
            assert (
                str("displacement_bc_magnitudes length") in exception.value)

    def test_displacement_bc_magnitudes_type(self, cython_model):
        """Test exception when displacement_bc_magnitudes type is wrong."""
        model = cython_model
        nlist_expected, n_neigh_expected = model.initial_connectivity
        steps = 10
        displacement_bc_magnitudes = 1.0
        with pytest.raises(TypeError) as exception:
            model._simulate_initialise(
                steps, 1, None, None, None, None,
                displacement_bc_magnitudes, None, None, None, None, None)
            assert (
                str("displacement_bc_magnitudes type") in exception.value)

    def test_force_bc_magnitudes_length(self, cython_model):
        """Test exception when force_bc_magnitudes length is wrong."""
        model = cython_model
        nlist_expected, n_neigh_expected = model.initial_connectivity
        steps = 10
        force_bc_magnitudes = np.zeros(9)
        with pytest.raises(ValueError) as exception:
            model._simulate_initialise(
                steps, 1, None, None, None, None,
                None, force_bc_magnitudes, None, None, None, None)
            assert (
                str("force_bc_magnitudes length") in exception.value)

    def test_force_bc_magnitudes_type(self, cython_model):
        """Test exception when force_bc_magnitudes type is wrong."""
        model = cython_model
        nlist_expected, n_neigh_expected = model.initial_connectivity
        steps = 10
        force_bc_magnitudes = 1.0
        with pytest.raises(TypeError) as exception:
            model._simulate_initialise(
                steps, 1, None, None, None, None,
                None, force_bc_magnitudes, None, None, None, None)
            assert (
                str("force_bc_magnitudes type") in exception.value)

    def test_connectivity_size(self, cython_model):
        """Test exception when connectivity size is wrong."""
        model = cython_model
        nlist_expected, n_neigh_expected = model.initial_connectivity
        steps = 10
        connectivity = (1, 2, 3)
        with pytest.raises(ValueError) as exception:
            model._simulate_initialise(
                steps, 1, None, None, None, None,
                None, None, connectivity, None, None, None)
            assert (
                str("connectivity size is wrong") in exception.value)

    def test_connectivity_type(self, cython_model):
        """Test exception when connectivity type is wrong."""
        model = cython_model
        nlist_expected, n_neigh_expected = model.initial_connectivity
        steps = 10
        connectivity = 1
        with pytest.raises(TypeError) as exception:
            model._simulate_initialise(
                steps, 1, None, None, None, None,
                None, None, connectivity, None, None, None)
            assert (
                str("connectivity type") in exception.value)

    def test_regimes_size(self, cython_model):
        """Test exception when regimes shape is wrong."""
        model = cython_model
        nlist_expected, n_neigh_expected = model.initial_connectivity
        steps = 10
        regimes = np.zeros((model.nnodes, model.max_neighbours - 1))
        with pytest.raises(ValueError) as exception:
            model._simulate_initialise(
                steps, 1, None, regimes, None, None,
                None, None, None, None, None, None)
            assert (
                str("regimes shape is wrong") in exception.value)

    def test_regimes_type(self, cython_model):
        """Test exception when regimes type is wrong."""
        model = cython_model
        nlist_expected, n_neigh_expected = model.initial_connectivity
        steps = 10
        regimes = 1
        with pytest.raises(TypeError) as exception:
            model._simulate_initialise(
                steps, 1, None, regimes, None, None,
                None, None, None, None, None, None)
            assert (
                str("regimes type") in exception.value)


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

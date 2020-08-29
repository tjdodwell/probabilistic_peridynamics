"""Tests for the integrators module."""
from .conftest import context_available
from ..integrators import (
    Integrator, Euler, EulerCL, EulerCromerCL, VelocityVerletCL, ContextError)
from ..model import Model, initial_crack_helper
from ..cl import get_context
import pytest
import numpy as np
import pyopencl as cl


@initial_crack_helper
def is_crack(x, y):
    """Determine whether a pair of particles define the crack."""
    crack_length = 0.3
    output = 0
    p1 = x
    p2 = y
    if x[0] > y[0]:
        p2 = x
        p1 = y
    # 1e-6 makes it fall one side of central line of particles
    if p1[0] < 0.5 + 1e-6 and p2[0] > 0.5 + 1e-6:
        # draw a straight line between them
        m = (p2[1] - p1[1]) / (p2[0] - p1[0])
        c = p1[1] - m * p1[0]
        # height a x = 0.5
        height = m * 0.5 + c
        if (height > 0.5 * (1 - crack_length)
                and height < 0.5 * (1 + crack_length)):
            output = 1
    return output


def is_density(x):
    """Return the density of the nodal volume."""
    return 1.0


@pytest.fixture(scope="module")
def euler_integrator(data_path, simple_displacement_boundary):
    """Run the example simulation on the Euler integrator."""
    path = data_path
    mesh_file = path / "example_mesh.vtk"
    euler = Euler(dt=1e-3)
    # Create model
    model = Model(mesh_file, integrator=euler, horizon=0.1,
                  critical_stretch=0.005,
                  bond_stiffness=18.0 * 0.05 / (np.pi * 0.1**4),
                  is_displacement_boundary=simple_displacement_boundary,
                  initial_crack=is_crack)

    return model, euler


@pytest.fixture(scope="module")
def euler_cl_integrator(data_path, simple_displacement_boundary):
    """Run the example simulation on the EulerCL integrator."""
    path = data_path
    mesh_file = path / "example_mesh.vtk"
    euler = EulerCL(dt=1e-3)
    # Create model
    model = Model(mesh_file, integrator=euler, horizon=0.1,
                  critical_stretch=0.005,
                  bond_stiffness=18.0 * 0.05 / (np.pi * 0.1**4),
                  is_displacement_boundary=simple_displacement_boundary,
                  initial_crack=is_crack)

    return model, euler


@pytest.fixture(scope="module")
def euler_cromer_cl_integrator(data_path, simple_displacement_boundary):
    """Run the example simulation on the EulerCromerCL integrator."""
    path = data_path
    mesh_file = path / "example_mesh.vtk"
    euler = EulerCromerCL(dt=1e-3, damping=0.0)
    # Create model
    model = Model(mesh_file, integrator=euler, horizon=0.1,
                  critical_stretch=0.005,
                  bond_stiffness=18.0 * 0.05 / (np.pi * 0.1**4),
                  is_displacement_boundary=simple_displacement_boundary,
                  initial_crack=is_crack,
                  is_density=is_density)

    return model, euler


@pytest.fixture(scope="module")
def velocity_verlet_cl_integrator(data_path, simple_displacement_boundary):
    """Run the example simulation on the VelocityVerletCL integrator =."""
    path = data_path
    mesh_file = path / "example_mesh.vtk"
    euler = VelocityVerletCL(dt=1e-3, damping=0.0)
    # Create model
    model = Model(mesh_file, integrator=euler, horizon=0.1,
                  critical_stretch=0.005,
                  bond_stiffness=18.0 * 0.05 / (np.pi * 0.1**4),
                  is_displacement_boundary=simple_displacement_boundary,
                  initial_crack=is_crack,
                  is_density=is_density)

    return model, euler


def test_no_context(data_path, monkeypatch):
    """Test raising error when no suitable device is found."""
    from .. import integrators
    # Mock the get_context function to return None as it would if no suitable
    # device is found.

    def return_none():
        return None
    monkeypatch.setattr(integrators, "get_context", return_none)
    with pytest.raises(ContextError) as exception:
        EulerCL(dt=1)

        assert "No suitable context was found." in exception.value


@context_available
def test_custom_context():
    """Test constructing an EulerCL object using the context argument."""
    context = get_context()
    integrator = EulerCL(dt=1, context=context)

    assert integrator.context is context


def test_invalid_custom_context():
    """Test constructing an EulerCL object using the context argument."""
    with pytest.raises(TypeError) as exception:
        EulerCL(dt=1, context=5)

        assert "context must be a pyopencl Context object" in exception.value


class TestIntegrator:
    """ABC class tests."""

    def test_not_implemented_error(self):
        """Ensure the ABC cannot be instantiated."""
        with pytest.raises(TypeError):
            Integrator(dt=1)


class TestEuler:
    """EulerCL integrator tests. See test_euler.py for more tests."""

    def test_call(self, data_path, euler_integrator):
        """Regression test for the Euler integrator."""
        path = data_path
        model, integrator = euler_integrator
        nlist, n_neigh = model.initial_connectivity
        u = np.zeros((model.nnodes, 3), dtype=np.float64)
        ud = np.zeros((model.nnodes, 3), dtype=np.float64)
        udd = np.zeros((model.nnodes, 3), dtype=np.float64)
        force = np.zeros((model.nnodes, 3), dtype=np.float64)
        body_force = np.zeros((model.nnodes, 3), dtype=np.float64)
        damage = np.zeros((model.nnodes, 3), dtype=np.float64)
        regimes = None

        integrator.create_buffers(
            nlist, n_neigh, model.bond_stiffness, model.critical_stretch,
            model.plus_cs, u, ud, udd, force, body_force, damage, regimes,
            model.nregimes, model.nbond_types)
        displacement_bc_magnitudes = 0.00001 / 2 * np.linspace(
            1, 10, 10)
        for step in range(10):
            integrator.__call__(
                displacement_bc_magnitude=displacement_bc_magnitudes[step],
                force_bc_magnitude=0.0)

        u_expected = np.load(path/"expected_displacements.npy")
        force_expected = np.load(path/"expected_force.npy")
        damage_expected = np.load(path/"expected_damage.npy")
        expected_connectivity = np.load(path/"expected_connectivity.npz")
        nlist_expected = expected_connectivity["nlist"]
        n_neigh_expected = expected_connectivity["n_neigh"]

        (u_actual,
         _,
         _,
         force_actual,
         _,
         damage_actual,
         nlist_actual,
         n_neigh_actual
         ) = integrator.write(
             u, ud, udd, force, body_force, damage, nlist, n_neigh)

        assert np.allclose(u_actual, u_expected)
        assert np.allclose(force_actual, force_expected)
        assert np.allclose(damage_actual, damage_expected)
        assert np.allclose(nlist_actual, nlist_expected)
        assert np.allclose(n_neigh_actual, n_neigh_expected)

    def test_create_buffers_nregimes(self, euler_integrator):
        """Test exception when n_regimes is supplied to Euler."""
        model, integrator = euler_integrator
        nlist, n_neigh = model.initial_connectivity
        bond_stiffness = model.bond_stiffness
        critical_stretch = model.critical_stretch
        plus_cs = model.plus_cs
        u = np.zeros((model.nnodes, 3), dtype=np.float64)
        ud = np.zeros((model.nnodes, 3), dtype=np.float64)
        udd = np.zeros((model.nnodes, 3), dtype=np.float64)
        force = np.zeros((model.nnodes, 3), dtype=np.float64)
        body_force = np.zeros((model.nnodes, 3), dtype=np.float64)
        damage = np.zeros((model.nnodes, 3), dtype=np.float64)
        regimes = None
        nregimes = 2
        nbond_types = 1

        with pytest.raises(ValueError) as exception:
            integrator.create_buffers(
                nlist, n_neigh, bond_stiffness, critical_stretch, plus_cs,
                u, ud, udd, force, body_force, damage, regimes, nregimes,
                nbond_types)
            assert (
                str("n-linear damage model's are not") in exception.value)

    def test_create_buffers_n_bond_types(self, euler_integrator):
        """Test exception when n_bond_types is supplied to Euler."""
        model, integrator = euler_integrator
        nlist, n_neigh = model.initial_connectivity
        bond_stiffness = model.bond_stiffness
        critical_stretch = model.critical_stretch
        plus_cs = model.plus_cs
        u = np.zeros((model.nnodes, 3), dtype=np.float64)
        ud = np.zeros((model.nnodes, 3), dtype=np.float64)
        udd = np.zeros((model.nnodes, 3), dtype=np.float64)
        force = np.zeros((model.nnodes, 3), dtype=np.float64)
        body_force = np.zeros((model.nnodes, 3), dtype=np.float64)
        damage = np.zeros((model.nnodes, 3), dtype=np.float64)
        regimes = None
        nregimes = 1
        nbond_types = 2

        with pytest.raises(ValueError) as exception:
            integrator.create_buffers(
                nlist, n_neigh, bond_stiffness, critical_stretch, plus_cs,
                u, ud, udd, force, body_force, damage, regimes, nregimes,
                nbond_types)
            assert (
                str("n-material composite models are not") in exception.value)

    def test_create_buffers(self, euler_integrator):
        """Test initiation of arrays that are dependent on simulation."""
        model, integrator = euler_integrator
        nlist, n_neigh = model.initial_connectivity
        bond_stiffness = model.bond_stiffness
        critical_stretch = model.critical_stretch
        plus_cs = model.plus_cs
        u = np.zeros((model.nnodes, 3), dtype=np.float64)
        ud = np.zeros((model.nnodes, 3), dtype=np.float64)
        udd = np.zeros((model.nnodes, 3), dtype=np.float64)
        force = np.zeros((model.nnodes, 3), dtype=np.float64)
        body_force = np.zeros((model.nnodes, 3), dtype=np.float64)
        damage = np.zeros((model.nnodes), dtype=np.float64)
        regimes = None
        nregimes = 1
        nbond_types = 1

        integrator.create_buffers(
            nlist, n_neigh, bond_stiffness, critical_stretch, plus_cs,
            u, ud, udd, force, body_force, damage, regimes, nregimes,
            nbond_types)

        assert np.allclose(integrator.nlist, nlist)
        assert np.allclose(integrator.n_neigh, n_neigh)
        assert np.allclose(integrator.bond_stiffness, bond_stiffness)
        assert np.allclose(integrator.critical_stretch, critical_stretch)
        assert np.allclose(integrator.u, u)
        assert np.allclose(integrator.ud, ud)
        assert np.allclose(integrator.udd, udd)
        assert np.allclose(integrator.force, force)
        assert np.allclose(integrator.body_force, body_force)

    def test_build(self, euler_integrator):
        """Test initiate integrator arrays."""
        model, integrator = euler_integrator

        nnodes = model.nnodes
        degrees_freedom = model.degrees_freedom
        max_neighbours = model.max_neighbours
        coords = model.coords
        family = model.family
        volume = model.volume
        bc_types = model.bc_types
        bc_values = model.bc_values
        force_bc_types = model.force_bc_types
        force_bc_values = model.force_bc_values
        stiffness_corrections = None
        bond_types = None
        densities = None

        integrator.build(
            nnodes, degrees_freedom, max_neighbours, coords,
            volume, family, bc_types, bc_values, force_bc_types,
            force_bc_values, stiffness_corrections, bond_types, densities)

        assert np.allclose(integrator.nnodes, nnodes)
        assert np.allclose(integrator.coords, coords)
        assert np.allclose(integrator.family, family)
        assert np.allclose(integrator.volume, volume)
        assert np.allclose(integrator.bc_types, bc_types)
        assert np.allclose(integrator.bc_values, bc_values)
        assert np.allclose(integrator.force_bc_types, force_bc_types)
        assert np.allclose(integrator.force_bc_values, force_bc_values)

    def test_build_exception(self, euler_integrator):
        """Test initiatiation of integrator arrays."""
        model, integrator = euler_integrator

        nnodes = model.nnodes
        degrees_freedom = model.degrees_freedom
        max_neighbours = model.max_neighbours
        coords = model.coords
        family = model.family
        volume = model.volume
        bc_types = model.bc_types
        bc_values = model.bc_values
        force_bc_types = model.force_bc_types
        force_bc_values = model.force_bc_values
        stiffness_corrections = None
        bond_types = 1
        densities = None

        with pytest.raises(ValueError) as exception:
            integrator.build(
                nnodes, degrees_freedom, max_neighbours, coords,
                volume, family, bc_types, bc_values, force_bc_types,
                force_bc_values, stiffness_corrections, bond_types, densities)
            assert (
                str("bond_types are not supported by this") in exception.value)

    def test_build_exception_stiffness_corrections(self, euler_integrator):
        """Test exception when stiffness_corrections are applied to Euler."""
        model, integrator = euler_integrator

        nnodes = model.nnodes
        degrees_freedom = model.degrees_freedom
        max_neighbours = model.max_neighbours
        coords = model.coords
        family = model.family
        volume = model.volume
        bc_types = model.bc_types
        bc_values = model.bc_values
        force_bc_types = model.force_bc_types
        force_bc_values = model.force_bc_values
        stiffness_corrections = 1
        bond_types = None
        densities = None

        with pytest.raises(ValueError) as exception:
            integrator.build(
                nnodes, degrees_freedom, max_neighbours, coords,
                volume, family, bc_types, bc_values, force_bc_types,
                force_bc_values, stiffness_corrections, bond_types, densities)
            assert (
                str("stiffness_corrections are not") in exception.value)

    def test_build_exception_densities(self, euler_integrator):
        """Test exception when densities are applied to Euler."""
        model, integrator = euler_integrator

        nnodes = model.nnodes
        degrees_freedom = model.degrees_freedom
        max_neighbours = model.max_neighbours
        coords = model.coords
        family = model.family
        volume = model.volume
        bc_types = model.bc_types
        bc_values = model.bc_values
        force_bc_types = model.force_bc_types
        force_bc_values = model.force_bc_values
        stiffness_corrections = None
        bond_types = None
        densities = 1

        with pytest.raises(ValueError) as exception:

            integrator.build(
                nnodes, degrees_freedom, max_neighbours, coords,
                volume, family, bc_types, bc_values, force_bc_types,
                force_bc_values, stiffness_corrections, bond_types, densities)
            assert (
                str("densities are not supported") in exception.value)

    def test_create_special_buffers(self, euler_integrator):
        """Test for no special buffers for this integrator."""
        model, integrator = euler_integrator
        value = integrator._create_special_buffers()
        assert value is None

    def test_build_special(self, euler_integrator):
        """There for no special programs for this integrator."""
        model, integrator = euler_integrator
        value = integrator._build_special()
        assert value is None


class TestEulerCL:
    """Euler integrator tests. See test_euler.py for more tests."""

    @context_available
    def test_call(self, data_path, euler_cl_integrator):
        """Regression test for the EulerCL integrator."""
        path = data_path
        model, integrator = euler_cl_integrator
        nlist, n_neigh = model.initial_connectivity
        u = np.zeros((model.nnodes, 3), dtype=np.float64)
        ud = np.zeros((model.nnodes, 3), dtype=np.float64)
        udd = np.zeros((model.nnodes, 3), dtype=np.float64)
        force = np.zeros((model.nnodes, 3), dtype=np.float64)
        body_force = np.zeros((model.nnodes, 3), dtype=np.float64)
        damage = np.zeros((model.nnodes), dtype=np.float64)
        regimes = None

        integrator.create_buffers(
            nlist, n_neigh, model.bond_stiffness, model.critical_stretch,
            model.plus_cs, u, ud, udd, force, body_force, damage, regimes,
            model.nregimes, model.nbond_types)
        displacement_bc_magnitudes = 0.00001 / 2 * np.linspace(
            1, 10, 10)
        for step in range(10):
            integrator.__call__(
                displacement_bc_magnitude=displacement_bc_magnitudes[step],
                force_bc_magnitude=0.0)

        u_expected = np.load(path/"expected_displacements.npy")
        force_expected = np.load(path/"expected_force.npy")
        damage_expected = np.load(path/"expected_damage.npy")
        expected_connectivity = np.load(path/"expected_connectivity_cl.npz")
        nlist_expected = expected_connectivity["nlist"]
        n_neigh_expected = expected_connectivity["n_neigh"]

        (u_actual,
         _,
         _,
         force_actual,
         _,
         damage_actual,
         nlist_actual,
         n_neigh_actual
         ) = integrator.write(
             u, ud, udd, force, body_force, damage, nlist, n_neigh)

        assert np.allclose(u_actual, u_expected)
        assert np.allclose(force_actual, force_expected)
        assert np.allclose(damage_actual, damage_expected)
        assert np.allclose(nlist_actual, nlist_expected)
        assert np.allclose(n_neigh_actual, n_neigh_expected)

    @context_available
    def test_create_buffers_float(self, euler_cl_integrator):
        """Test initiation of arrays that are dependent on simulation."""
        model, integrator = euler_cl_integrator
        nlist, n_neigh = model.initial_connectivity
        bond_stiffness = model.bond_stiffness
        critical_stretch = model.critical_stretch
        plus_cs = model.plus_cs
        u = np.zeros((model.nnodes, 3), dtype=np.float64)
        ud = np.zeros((model.nnodes, 3), dtype=np.float64)
        udd = np.zeros((model.nnodes, 3), dtype=np.float64)
        force = np.zeros((model.nnodes, 3), dtype=np.float64)
        body_force = np.zeros((model.nnodes, 3), dtype=np.float64)
        damage = np.zeros((model.nnodes, 3), dtype=np.float64)
        regimes = None
        nregimes = 1
        nbond_types = 1

        integrator.create_buffers(
            nlist, n_neigh, bond_stiffness, critical_stretch, plus_cs,
            u, ud, udd, force, body_force, damage, regimes, nregimes,
            nbond_types)

        assert(type(integrator.bond_stiffness_d) is np.float64)
        assert(type(integrator.critical_stretch_d) is np.float64)

    @context_available
    def test_create_buffers_array(self, euler_cl_integrator):
        """Test initiation of arrays that are dependent on simulation."""
        model, integrator = euler_cl_integrator
        nlist, n_neigh = model.initial_connectivity
        bond_stiffness = model.bond_stiffness
        critical_stretch = model.critical_stretch
        plus_cs = np.zeros((2, 2), dtype=np.float64)
        u = np.zeros((model.nnodes, 3), dtype=np.float64)
        ud = np.zeros((model.nnodes, 3), dtype=np.float64)
        udd = np.zeros((model.nnodes, 3), dtype=np.float64)
        force = np.zeros((model.nnodes, 3), dtype=np.float64)
        body_force = np.zeros((model.nnodes, 3), dtype=np.float64)
        damage = np.zeros((model.nnodes, 3), dtype=np.float64)
        regimes = np.zeros(
            (model.nnodes, model.max_neighbours), dtype=np.float64)
        nregimes = 2
        nbond_types = 2

        integrator.create_buffers(
            nlist, n_neigh, bond_stiffness, critical_stretch, plus_cs,
            u, ud, udd, force, body_force, damage, regimes, nregimes,
            nbond_types)

        assert(type(integrator.bond_stiffness_d) is cl._cl.Buffer)
        assert(type(integrator.critical_stretch_d) is cl._cl.Buffer)

    @context_available
    def test_build_exception(self, euler_cl_integrator):
        """Test exception when densities are supplied to EulerCL."""
        model, integrator = euler_cl_integrator

        nnodes = model.nnodes
        degrees_freedom = model.degrees_freedom
        max_neighbours = model.max_neighbours
        coords = model.coords
        family = model.family
        volume = model.volume
        bc_types = model.bc_types
        bc_values = model.bc_values
        force_bc_types = model.force_bc_types
        force_bc_values = model.force_bc_values
        stiffness_corrections = None
        bond_types = None
        densities = 1

        with pytest.raises(ValueError) as exception:

            integrator.build(
                nnodes, degrees_freedom, max_neighbours, coords,
                volume, family, bc_types, bc_values, force_bc_types,
                force_bc_values, stiffness_corrections, bond_types, densities)
            assert (
                str("densities are not supported") in exception.value)

    @context_available
    def test_create_special_buffers(self, euler_cl_integrator):
        """There are no special buffers so this method does nothing."""
        model, integrator = euler_cl_integrator
        value = integrator._create_special_buffers()
        assert value is None


class TestEulerCromerCL:
    """EulerCromerCL integrator tests. See test_euler.py for more tests."""

    @context_available
    def test_call(self, data_path, euler_cromer_cl_integrator):
        """Regression test for the EulerCromerCL integrator."""
        path = data_path
        model, integrator = euler_cromer_cl_integrator
        nlist, n_neigh = model.initial_connectivity
        u = np.zeros((model.nnodes, 3), dtype=np.float64)
        ud = np.zeros((model.nnodes, 3), dtype=np.float64)
        udd = np.zeros((model.nnodes, 3), dtype=np.float64)
        force = np.zeros((model.nnodes, 3), dtype=np.float64)
        body_force = np.zeros((model.nnodes, 3), dtype=np.float64)
        damage = np.zeros((model.nnodes), dtype=np.float64)
        regimes = None

        integrator.create_buffers(
            nlist, n_neigh, model.bond_stiffness, model.critical_stretch,
            model.plus_cs, u, ud, udd, force, body_force, damage, regimes,
            model.nregimes, model.nbond_types)
        displacement_bc_magnitudes = 0.00001 / 2 * np.linspace(
            1, 10, 10)
        for step in range(10):
            integrator.__call__(
                displacement_bc_magnitude=displacement_bc_magnitudes[step],
                force_bc_magnitude=0.0)

        (u_actual,
         ud_actual,
         _,
         force_actual,
         _,
         damage_actual,
         nlist_actual,
         n_neigh_actual
         ) = integrator.write(
             u, ud, udd, force, body_force, damage, nlist, n_neigh)

        u_expected = np.load(path/"expected_displacements_euler_cromer.npy")
        force_expected = np.load(path/"expected_force_euler_cromer.npy")
        damage_expected = np.load(path/"expected_damage_euler_cromer.npy")
        expected_connectivity = np.load(
            path/"expected_connectivity_euler_cromer_cl.npz")
        nlist_expected = expected_connectivity["nlist"]
        n_neigh_expected = expected_connectivity["n_neigh"]

        assert np.allclose(u_actual, u_expected)
        assert np.allclose(force_actual, force_expected)
        assert np.allclose(damage_actual, damage_expected)
        assert np.allclose(nlist_actual, nlist_expected)
        assert np.allclose(n_neigh_actual, n_neigh_expected)

    @context_available
    def test_create_buffers_float(self, euler_cromer_cl_integrator):
        """Test initiation of arrays that are dependent on simulation."""
        model, integrator = euler_cromer_cl_integrator
        nlist, n_neigh = model.initial_connectivity
        bond_stiffness = model.bond_stiffness
        critical_stretch = model.critical_stretch
        plus_cs = model.plus_cs
        u = np.zeros((model.nnodes, 3), dtype=np.float64)
        ud = np.zeros((model.nnodes, 3), dtype=np.float64)
        udd = np.zeros((model.nnodes, 3), dtype=np.float64)
        force = np.zeros((model.nnodes, 3), dtype=np.float64)
        body_force = np.zeros((model.nnodes, 3), dtype=np.float64)
        damage = np.zeros((model.nnodes, 3), dtype=np.float64)
        regimes = None
        nregimes = 1
        nbond_types = 1

        integrator.create_buffers(
            nlist, n_neigh, bond_stiffness, critical_stretch, plus_cs,
            u, ud, udd, force, body_force, damage, regimes, nregimes,
            nbond_types)

        assert(type(integrator.bond_stiffness_d) is np.float64)
        assert(type(integrator.critical_stretch_d) is np.float64)

    @context_available
    def test_create_buffers_array(self, euler_cromer_cl_integrator):
        """Test initiation of arrays that are dependent on simulation."""
        model, integrator = euler_cromer_cl_integrator
        nlist, n_neigh = model.initial_connectivity
        bond_stiffness = model.bond_stiffness
        critical_stretch = model.critical_stretch
        plus_cs = np.zeros((2, 2), dtype=np.float64)
        u = np.zeros((model.nnodes, 3), dtype=np.float64)
        ud = np.zeros((model.nnodes, 3), dtype=np.float64)
        udd = np.zeros((model.nnodes, 3), dtype=np.float64)
        force = np.zeros((model.nnodes, 3), dtype=np.float64)
        body_force = np.zeros((model.nnodes, 3), dtype=np.float64)
        damage = np.zeros((model.nnodes, 3), dtype=np.float64)
        regimes = np.zeros(
            (model.nnodes, model.max_neighbours), dtype=np.float64)
        nregimes = 2
        nbond_types = 2

        integrator.create_buffers(
            nlist, n_neigh, bond_stiffness, critical_stretch, plus_cs,
            u, ud, udd, force, body_force, damage, regimes, nregimes,
            nbond_types)

        assert(type(integrator.bond_stiffness_d) is cl._cl.Buffer)
        assert(type(integrator.critical_stretch_d) is cl._cl.Buffer)

    @context_available
    def test_build_exception(self, euler_cromer_cl_integrator):
        """Test exception when densities are not supplied to EulerCromerCL."""
        model, integrator = euler_cromer_cl_integrator

        nnodes = model.nnodes
        degrees_freedom = model.degrees_freedom
        max_neighbours = model.max_neighbours
        coords = model.coords
        family = model.family
        volume = model.volume
        bc_types = model.bc_types
        bc_values = model.bc_values
        force_bc_types = model.force_bc_types
        force_bc_values = model.force_bc_values
        stiffness_corrections = None
        bond_types = None
        densities = None

        with pytest.raises(ValueError) as exception:

            integrator.build(
                nnodes, degrees_freedom, max_neighbours, coords,
                volume, family, bc_types, bc_values, force_bc_types,
                force_bc_values, stiffness_corrections, bond_types, densities)
            assert (
                str("densities must be supplied") in exception.value)

    @context_available
    def test_create_special_buffers(self, euler_cromer_cl_integrator):
        """There are no special buffers so this method does nothing."""
        model, integrator = euler_cromer_cl_integrator
        value = integrator._create_special_buffers()
        assert value is None


class TestVelocityVerletCL:
    """VelocityVerletCL integrator tests. See test_euler.py for more tests."""

    @context_available
    def test_call(self, data_path, velocity_verlet_cl_integrator):
        """Regression test for the VelocityVerletCL integrator."""
        path = data_path
        model, integrator = velocity_verlet_cl_integrator
        nlist, n_neigh = model.initial_connectivity
        u = np.zeros((model.nnodes, 3), dtype=np.float64)
        ud = np.zeros((model.nnodes, 3), dtype=np.float64)
        udd = np.zeros((model.nnodes, 3), dtype=np.float64)
        force = np.zeros((model.nnodes, 3), dtype=np.float64)
        body_force = np.zeros((model.nnodes, 3), dtype=np.float64)
        damage = np.zeros((model.nnodes), dtype=np.float64)
        regimes = None

        integrator.create_buffers(
            nlist, n_neigh, model.bond_stiffness, model.critical_stretch,
            model.plus_cs, u, ud, udd, force, body_force, damage, regimes,
            model.nregimes, model.nbond_types)
        displacement_bc_magnitudes = 0.00001 / 2 * np.linspace(
            1, 10, 10)
        for step in range(10):
            integrator.__call__(
                displacement_bc_magnitude=displacement_bc_magnitudes[step],
                force_bc_magnitude=0.0)

        (u_actual,
         ud_actual,
         udd_actual,
         force_actual,
         _,
         damage_actual,
         nlist_actual,
         n_neigh_actual
         ) = integrator.write(
             u, ud, udd, force, body_force, damage, nlist, n_neigh)

        u_expected = np.load(path/"expected_displacements_velocity_verlet.npy")
        ud_expected = np.load(path/"expected_velocities_velocity_verlet.npy")
        udd_expected = np.load(
            path/"expected_accelerations_velocity_verlet.npy")
        force_expected = np.load(path/"expected_force_velocity_verlet.npy")
        damage_expected = np.load(path/"expected_damage_velocity_verlet.npy")
        expected_connectivity = np.load(
            path/"expected_connectivity_velocity_verlet_cl.npz")
        nlist_expected = expected_connectivity["nlist"]
        n_neigh_expected = expected_connectivity["n_neigh"]

        assert np.allclose(u_actual, u_expected)
        assert np.allclose(ud_actual, ud_expected)
        assert np.allclose(udd_actual, udd_expected)
        assert np.allclose(force_actual, force_expected)
        assert np.allclose(damage_actual, damage_expected)
        assert np.allclose(nlist_actual, nlist_expected)
        assert np.allclose(n_neigh_actual, n_neigh_expected)

    @context_available
    def test_create_buffers_float(self, velocity_verlet_cl_integrator):
        """Test initiation of arrays that are dependent on simulation."""
        model, integrator = velocity_verlet_cl_integrator
        nlist, n_neigh = model.initial_connectivity
        bond_stiffness = model.bond_stiffness
        critical_stretch = model.critical_stretch
        plus_cs = model.plus_cs
        u = np.zeros((model.nnodes, 3), dtype=np.float64)
        ud = np.zeros((model.nnodes, 3), dtype=np.float64)
        udd = np.zeros((model.nnodes, 3), dtype=np.float64)
        force = np.zeros((model.nnodes, 3), dtype=np.float64)
        body_force = np.zeros((model.nnodes, 3), dtype=np.float64)
        damage = np.zeros((model.nnodes, 3), dtype=np.float64)
        regimes = None
        nregimes = 1
        nbond_types = 1

        integrator.create_buffers(
            nlist, n_neigh, bond_stiffness, critical_stretch, plus_cs,
            u, ud, udd, force, body_force, damage, regimes, nregimes,
            nbond_types)

        assert(type(integrator.bond_stiffness_d) is np.float64)
        assert(type(integrator.critical_stretch_d) is np.float64)

    @context_available
    def test_create_buffers_array(self, velocity_verlet_cl_integrator):
        """Test initiation of arrays that are dependent on simulation."""
        model, integrator = velocity_verlet_cl_integrator
        nlist, n_neigh = model.initial_connectivity
        bond_stiffness = model.bond_stiffness
        critical_stretch = model.critical_stretch
        plus_cs = np.zeros((2, 2), dtype=np.float64)
        u = np.zeros((model.nnodes, 3), dtype=np.float64)
        ud = np.zeros((model.nnodes, 3), dtype=np.float64)
        udd = np.zeros((model.nnodes, 3), dtype=np.float64)
        force = np.zeros((model.nnodes, 3), dtype=np.float64)
        body_force = np.zeros((model.nnodes, 3), dtype=np.float64)
        damage = np.zeros((model.nnodes, 3), dtype=np.float64)
        regimes = np.zeros(
            (model.nnodes, model.max_neighbours), dtype=np.float64)
        nregimes = 2
        nbond_types = 2

        integrator.create_buffers(
            nlist, n_neigh, bond_stiffness, critical_stretch, plus_cs,
            u, ud, udd, force, body_force, damage, regimes, nregimes,
            nbond_types)

        assert(type(integrator.bond_stiffness_d) is cl._cl.Buffer)
        assert(type(integrator.critical_stretch_d) is cl._cl.Buffer)

    @context_available
    def test_build_exception(self, velocity_verlet_cl_integrator):
        """Test exception when densities not supplied to VelocityVerletCL."""
        model, integrator = velocity_verlet_cl_integrator

        nnodes = model.nnodes
        degrees_freedom = model.degrees_freedom
        max_neighbours = model.max_neighbours
        coords = model.coords
        family = model.family
        volume = model.volume
        bc_types = model.bc_types
        bc_values = model.bc_values
        force_bc_types = model.force_bc_types
        force_bc_values = model.force_bc_values
        stiffness_corrections = None
        bond_types = None
        densities = None

        with pytest.raises(ValueError) as exception:

            integrator.build(
                nnodes, degrees_freedom, max_neighbours, coords,
                volume, family, bc_types, bc_values, force_bc_types,
                force_bc_values, stiffness_corrections, bond_types, densities)
            assert (
                str("densities must be supplied") in exception.value)

    @context_available
    def test_create_special_buffers(self, velocity_verlet_cl_integrator):
        """There are no special buffers so this method does nothing."""
        model, integrator = velocity_verlet_cl_integrator
        value = integrator._create_special_buffers()
        assert value is None

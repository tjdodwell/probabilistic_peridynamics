"""Test the ModelCL class."""
from .conftest import context_available
from ..model_cl import ModelCL, ContextError
from ..cl import get_context
import numpy as np
import pyopencl as cl
from pyopencl import mem_flags as mf
import pytest


@pytest.fixture
def basic_model_2d(data_path):
    """Create a basic 2D model object."""
    mesh_file = data_path / "example_mesh.vtk"
    model = ModelCL(mesh_file, horizon=0.1, critical_stretch=0.05,
                    bond_stiffness=18.0 * 0.05 / (np.pi * 0.1**4))
    return model


@pytest.fixture()
def basic_model_3d(data_path):
    """Create a basic 3D model object."""
    mesh_file = data_path / "example_mesh_3d.vtk"
    model = ModelCL(mesh_file, horizon=0.1, critical_stretch=0.05,
                    bond_stiffness=18.0 * 0.05 / (np.pi * 0.1**4),
                    dimensions=3)
    return model


def test_no_context(data_path, monkeypatch):
    """Test raising error when no suitable device is found."""
    from .. import model_cl

    # Mock the get_context function to return None as it would if no suitable
    # device is found.
    def return_none():
        return None
    monkeypatch.setattr(model_cl, "get_context", return_none)

    mesh_file = data_path / "example_mesh.vtk"
    with pytest.raises(ContextError) as exception:
        ModelCL(mesh_file, horizon=0.1, critical_stretch=0.05,
                bond_stiffness=18.0 * 0.05 / (np.pi * 0.1**4))

        assert "No suitable context was found." in exception.value


@context_available
def test_custom_context(data_path):
    """Test constructing a ModelCL object using the context argument."""
    mesh_file = data_path / "example_mesh_3d.vtk"
    context = get_context()
    model = ModelCL(mesh_file, horizon=0.1, critical_stretch=0.05,
                    bond_stiffness=18.0 * 0.05 / (np.pi * 0.1**4),
                    dimensions=3, context=context)

    assert model.context is context


def test_invalid_custom_context(data_path):
    """Test constructing a ModelCL object using the context argument."""
    mesh_file = data_path / "example_mesh_3d.vtk"
    with pytest.raises(TypeError) as exception:
        ModelCL(mesh_file, horizon=0.1, critical_stretch=0.05,
                bond_stiffness=18.0 * 0.05 / (np.pi * 0.1**4),
                dimensions=3, context=5)

        assert "context must be a pyopencl Context object" in exception.value


def test_initial_damage_2d(basic_model_2d):
    """Ensure initial damage is zero."""
    model = basic_model_2d
    context = model.context
    queue = model.queue
    nlist, n_neigh = model.initial_connectivity

    n_neigh_d = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR,
                          hostbuf=n_neigh)
    family_d = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR,
                         hostbuf=model.family)
    damage = np.empty(n_neigh.shape, dtype=np.float64)
    damage_d = cl.Buffer(context, mf.WRITE_ONLY, damage.nbytes)

    model._damage(n_neigh_d, family_d, damage_d)
    cl.enqueue_copy(queue, damage, damage_d)

    assert np.all(damage == 0)


def test_initial_damage_3d(basic_model_3d):
    """Ensure initial damage is zero."""
    model = basic_model_3d
    context = model.context
    queue = model.queue
    nlist, n_neigh = model.initial_connectivity

    n_neigh_d = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR,
                          hostbuf=n_neigh)
    family_d = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR,
                         hostbuf=model.family)
    damage = np.empty(n_neigh.shape, dtype=np.float64)
    damage_d = cl.Buffer(context, mf.WRITE_ONLY, damage.nbytes)

    model._damage(n_neigh_d, family_d, damage_d)
    cl.enqueue_copy(queue, damage, damage_d)

    assert np.all(damage == 0)

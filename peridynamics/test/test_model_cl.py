"""Test the ModelCL class."""
from .conftest import context_available
from ..model_cl import ModelCL, ContextError
from ..cl import get_context
import pytest


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
        ModelCL(mesh_file, horizon=0.1, critical_strain=0.05,
                elastic_modulus=0.05)

        assert "No suitable context was found." in exception.value


@context_available
def test_custom_context(data_path):
    """Test constructing a ModelCL object using the context argument."""
    mesh_file = data_path / "example_mesh_3d.vtk"
    context = get_context()
    model = ModelCL(mesh_file, horizon=0.1, critical_strain=0.05,
                    elastic_modulus=0.05, dimensions=3, context=context)

    assert model.context is context


def test_invalid_custom_context(data_path):
    """Test constructing a ModelCL object using the context argument."""
    mesh_file = data_path / "example_mesh_3d.vtk"
    with pytest.raises(TypeError):
        ModelCL(mesh_file, horizon=0.1, critical_strain=0.05,
                elastic_modulus=0.05, dimensions=3, context=5)

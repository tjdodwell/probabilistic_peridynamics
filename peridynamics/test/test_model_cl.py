"""Test the ModelCL class."""
from ..model_cl import ModelCL, ContextError
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

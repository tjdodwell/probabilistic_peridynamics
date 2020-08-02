"""Tests for the integrators module."""
from .conftest import context_available
from ..integrators import Integrator, EulerCL, ContextError
from ..cl import get_context
import pytest


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

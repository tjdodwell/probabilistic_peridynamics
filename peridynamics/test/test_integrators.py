"""Tests for the integrators module."""
from ..integrators import Integrator
import pytest


class TestIntegrator:
    """ABC class tests."""

    def test_not_implemented_error(self):
        """Ensure the ABC cannot be instantiated."""
        with pytest.raises(TypeError):
            Integrator(dt=1)

"""
Tests for the integrators module.
"""
from ..integrators import Euler
import numpy as np


class TestEuler(object):
    """
    Euler integrator tests.
    """
    def test_basic_integration(self):
        integrator = Euler(1)
        u = np.zeros(3)
        f = np.array([1.0, 2.0, 3.0])

        u = integrator.step(u, f)

        assert np.all(u == f)

    def test_basic_integration2(self):
        integrator = Euler(2.0)
        u = np.zeros(3)
        f = np.array([1.0, 2.0, 3.0])

        u = integrator.step(u, f)

        assert np.all(u == 2.0*f)

    def test_basic_integration3(self):
        integrator = Euler(2.0, dampening=0.7)
        u = np.zeros(3)
        f = np.array([1.0, 2.0, 3.0])

        u = integrator.step(u, f)

        assert np.all(u == 2.0*0.7*f)

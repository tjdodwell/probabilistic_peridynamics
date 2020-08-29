"""Tests for the spatial module."""
import numpy as np
from peripy.spatial import euclid, strain, strain2
from scipy.spatial.distance import euclidean


class TestEuclid():
    """Test euclidean distance function."""

    def test_euclid1(self):
        """Ensure results are consistent with scipy."""
        r1 = np.random.random(3)
        r2 = np.random.random(3)

        assert np.allclose(euclidean(r1, r2), euclid(r1, r2))

    def test_euclid2(self):
        """Test negative numbers."""
        r1 = np.random.random(3)
        r2 = -np.random.random(3)

        assert np.allclose(euclidean(r1, r2), euclid(r1, r2))


class TestStrain():
    """Test the strain function."""

    def test_strain1(self):
        """Ensure function is consistent with scipy."""
        r10 = np.random.random(3)
        r20 = np.random.random(3)
        l0 = euclidean(r10, r20)

        r1 = r10 + np.random.random(3)*0.1
        r2 = r20 + np.random.random(3)*0.1

        strain_actual = strain(r1, r2, r10, r20)
        strain_expected = (euclidean(r1, r2) - l0)/l0
        assert np.isclose(strain_actual, strain_expected)

        strain2_actual = strain2(euclidean(r1, r2), r10, r20)
        assert np.isclose(strain2_actual, strain_expected)

    def test_strain2(self):
        """Test a trivial, known example."""
        r10 = np.array([0.0, 0.0, 0.0])
        r20 = np.array([1.0, 0.0, 0.0])

        r1 = np.array([0.0, 0.0, 0.0])
        r2 = np.array([2.0, 0.0, 0.0])

        assert np.isclose(strain(r1, r2, r10, r20), 1.0)

        assert np.isclose(strain2(euclidean(r1, r2), r10, r20), 1.0)

    def test_strain3(self):
        """Test a trivial, known example."""
        r10 = np.array([0.0, 0.0, 0.0])
        r20 = np.array([1.0, 0.0, 0.0])

        r1 = r10
        r2 = r20

        assert np.isclose(strain(r1, r2, r10, r20), 0.0)

        assert np.isclose(strain2(euclidean(r1, r2), r10, r20), 0.0)

    def test_strain4(self):
        """Test a trivial, known example."""
        r10 = np.array([0.0, 0.0, 0.0])
        r20 = np.array([2.0, 0.0, 0.0])

        r1 = np.array([0.0, 0.0, 0.0])
        r2 = np.array([3.0, 0.0, 0.0])

        assert np.isclose(strain(r1, r2, r10, r20), 0.5)

        assert np.isclose(strain2(euclidean(r1, r2), r10, r20), 0.5)

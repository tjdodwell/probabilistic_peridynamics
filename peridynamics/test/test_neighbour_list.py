"""Tests for the neighbour list module."""
from peridynamics.neighbour_list import (
    euclid, strain, family, create_neighbour_list, break_bonds
    )
import numpy as np
from scipy.spatial.distance import euclidean, cdist


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
        assert np.allclose(strain_actual, strain_expected)

    def test_strain2(self):
        """Test a trivial, known example."""
        r10 = np.array([0.0, 0.0, 0.0])
        r20 = np.array([1.0, 0.0, 0.0])

        r1 = np.array([0.0, 0.0, 0.0])
        r2 = np.array([2.0, 0.0, 0.0])

        assert np.isclose(strain(r1, r2, r10, r20), 1.0)

    def test_strain3(self):
        """Test a trivial, known example."""
        r10 = np.array([0.0, 0.0, 0.0])
        r20 = np.array([1.0, 0.0, 0.0])

        r1 = r10
        r2 = r20

        assert np.isclose(strain(r1, r2, r10, r20), 0.0)

    def test_strain4(self):
        """Test a trivial, known example."""
        r10 = np.array([0.0, 0.0, 0.0])
        r20 = np.array([2.0, 0.0, 0.0])

        r1 = np.array([0.0, 0.0, 0.0])
        r2 = np.array([3.0, 0.0, 0.0])

        assert np.isclose(strain(r1, r2, r10, r20), 0.5)


def test_family():
    """Test family function."""
    r = np.random.random((100, 3))
    horizon = 0.2

    family_actual = family(r, horizon)
    family_expected = np.sum(cdist(r, r) < horizon, axis=0) - 1

    assert np.all(family_actual == family_expected)


def test_neighbour_list():
    """Test neighbour list function."""
    r = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [2.0, 0.0, 0.0]
        ])

    nl, n_neigh = create_neighbour_list(r, 1.1, 3)
    nl_expected = np.array([
        [1, 2, 0],
        [0, 3, 0],
        [0, 0, 0],
        [1, 0, 0]
        ])
    n_neigh_expected = np.array([2, 2, 1, 1])

    assert np.all(nl == nl_expected)
    assert np.all(n_neigh == n_neigh_expected)


def test_break_bonds():
    """Test neighbour list function."""
    r0 = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [2.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        ])
    horizon = 1.1
    nl, n_neigh = create_neighbour_list(r0, horizon, 3)

    nl_expected = np.array([
        [1, 2, 4],
        [0, 3, 0],
        [0, 0, 0],
        [1, 0, 0],
        [0, 0, 0]
        ])
    n_neigh_expected = np.array([3, 2, 1, 1, 1])

    assert np.all(nl == nl_expected)
    assert np.all(n_neigh == n_neigh_expected)

    r = np.array([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [3.0, 0.0, 0.0],
        [0.0, 0.0, 2.0],
        ])
    critical_strain = 1.0

    break_bonds(r, r0, nl, n_neigh, critical_strain)
    nl_expected = np.array([
        [2, 2, 4],
        [3, 3, 0],
        [0, 0, 0],
        [1, 0, 0],
        [0, 0, 0]
        ])
    n_neigh_expected = np.array([1, 1, 1, 1, 0])

    assert np.all(nl == nl_expected)
    assert np.all(n_neigh == n_neigh_expected)

"""Tests for the neighbour list module."""
from peridynamics.neighbour_list import euclid, family, create_neighbour_list
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


def test_family():
    """Test family function."""
    r = np.random.random((100, 3))
    horizon = 0.2

    family_actual = family(r, horizon)
    family_expected = np.sum(cdist(r, r) < horizon, axis=0)

    assert np.all(family_actual == family_expected)


def test_neighbour_list():
    """Test neighbour list function."""
    r = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [2.0, 0.0, 0.0]
        ])

    nl = create_neighbour_list(r, 1.1, 3)
    nl_expected = np.array([
        [1, 2, 0],
        [0, 3, 0],
        [0, 0, 0],
        [1, 0, 0]
        ])

    print(nl)
    assert np.all(nl == nl_expected)

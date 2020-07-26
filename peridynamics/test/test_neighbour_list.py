"""Tests for the neighbour list module."""
from peridynamics.neighbour_list import (
    set_family, create_neighbour_list, break_bonds, create_crack
    )
import numpy as np
from scipy.spatial.distance import cdist


def test_family():
    """Test family function."""
    r = np.random.random((100, 3))
    horizon = 0.2

    family_actual = set_family(r, horizon)
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


def test_create_crack():
    """Test crack creations function."""
    r = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [2.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        ])
    horizon = 1.1
    nl, n_neigh = create_neighbour_list(r, horizon, 3)

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

    crack = np.array([(0, 2), (1, 3)], dtype=np.int32)
    create_crack(crack, nl, n_neigh)

    nl_expected = np.array([
        [1, 4, 4],
        [0, 3, 0],
        [0, 0, 0],
        [1, 0, 0],
        [0, 0, 0]
        ])
    n_neigh_expected = np.array([2, 1, 0, 0, 1])
    assert np.all(nl == nl_expected)
    assert np.all(n_neigh == n_neigh_expected)

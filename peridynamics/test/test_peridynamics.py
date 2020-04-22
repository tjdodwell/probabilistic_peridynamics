"""Tests for the peridynamics modules."""
import numpy as np
from peridynamics.neighbour_list import create_neighbour_list
from peridynamics.peridynamics import damage, bond_force


def test_damage():
    """Test damage function."""
    family = np.array([10, 5, 5, 1, 5, 7, 10, 3, 3, 4], dtype=np.int32)
    n_neigh = np.array([5, 5, 3, 0, 4, 5, 8, 3, 2, 1], dtype=np.int32)

    damage_actual = damage(n_neigh, family)
    damage_expected = (family - n_neigh) / family

    assert np.allclose(damage_actual, damage_expected)


class TestForce():
    """Test force calculation."""

    def test_initial_force(self):
        """Ensure forces are zero when there is no displacement."""
        r0 = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            ])
        horizon = 1.1
        volume = np.ones(5)
        bond_stiffness = 1.0
        nl, n_neigh = create_neighbour_list(r0, horizon, 3)

        force_expected = np.zeros((5, 3))
        force_actual = bond_force(r0, r0, nl, n_neigh, volume, bond_stiffness)

        assert np.allclose(force_actual, force_expected)

    def test_force(self):
        """Ensure forces are in the correct direction using a minimal model."""
        r0 = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            ])
        horizon = 1.01
        elastic_modulus = 0.05
        bond_stiffness = 18.0 * elastic_modulus / (np.pi * horizon**4)
        volume = np.full(3, 0.16666667)
        nl, n_neigh = create_neighbour_list(r0, horizon, 3)

        # Displace particles, but do not update neighbour list
        r = r0 + np.array([
            [0.0, 0.0, 0.0],
            [0.05, 0.0, 0.0],
            [0.05, 0.05, 0.0]
            ])

        actual_force = bond_force(r, r0, nl, n_neigh, volume,
                                  bond_stiffness)

        # Ensure force array is correct
        force_value = 0.00229417
        expected_force = np.array([
            [force_value, 0., 0.],
            [-force_value, force_value, 0.],
            [0., -force_value, 0.]
            ])
        assert np.allclose(actual_force, expected_force)

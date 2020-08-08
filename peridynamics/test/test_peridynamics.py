"""Tests for the peridynamics modules."""
import numpy as np
from peridynamics.neighbour_list import create_neighbour_list_cython
from peridynamics.peridynamics import (damage, bond_force, break_bonds,
                                       update_displacement)


def test_damage():
    """Test damage function."""
    family = np.array([10, 5, 5, 1, 5, 7, 10, 3, 3, 4], dtype=np.int32)
    n_neigh = np.array([5, 5, 3, 0, 4, 5, 8, 3, 2, 1], dtype=np.int32)

    damage_actual = damage(n_neigh, family)
    damage_expected = (family - n_neigh) / family

    assert np.allclose(damage_actual, damage_expected)


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
    nl, n_neigh = create_neighbour_list_cython(r0, horizon, 3)

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


class TestForce:
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
        nnodes = 5
        volume = np.ones(nnodes)
        bond_stiffness = 1.0
        nl, n_neigh = create_neighbour_list_cython(r0, horizon, 3)
        force_bc_scale = 1.0
        force_bc_types = np.zeros((nnodes, 3), dtype=np.int32)
        force_bc_values = np.zeros((nnodes, 3), dtype=np.float64)

        force_expected = np.zeros((5, 3))
        force_actual = bond_force(r0, r0, nl, n_neigh, volume, bond_stiffness,
                                  force_bc_values, force_bc_types,
                                  force_bc_scale)
        assert np.allclose(force_actual, force_expected)

    def test_force(self):
        """Ensure forces are in the correct direction using a minimal model."""
        r0 = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            ])
        horizon = 1.01
        nnodes = 3
        elastic_modulus = 0.05
        bond_stiffness = 18.0 * elastic_modulus / (np.pi * horizon**4)
        volume = np.full(nnodes, 0.16666667)
        nl, n_neigh = create_neighbour_list_cython(r0, horizon, 3)
        force_bc_scale = 1.0
        force_bc_types = np.zeros((nnodes, 3), dtype=np.int32)
        force_bc_values = np.zeros((nnodes, 3), dtype=np.float64)

        # Displace particles, but do not update neighbour list
        r = r0 + np.array([
            [0.0, 0.0, 0.0],
            [0.05, 0.0, 0.0],
            [0.05, 0.05, 0.0]
            ])

        actual_force = bond_force(r, r0, nl, n_neigh, volume,
                                  bond_stiffness, force_bc_values,
                                  force_bc_types, force_bc_scale)

        # Ensure force array is correct
        force_value = 0.00229417
        expected_force = np.array([
            [force_value, 0., 0.],
            [-force_value, force_value, 0.],
            [0., -force_value, 0.]
            ])
        assert np.allclose(actual_force, expected_force)


class TestUpdateDisplacement:
    """Test the displacement update."""

    def test_update_displacement(self):
        """Test basic displacement update."""
        nnodes = 3
        u = np.zeros((nnodes, 3))
        f = np.array([
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0]])
        bc_types = np.zeros((nnodes, 3), dtype=np.int32)
        bc_values = np.zeros((nnodes, 3))
        bc_scale = 0
        dt = 1.0
        update_displacement(u, bc_values, bc_types, f, bc_scale, dt)
        assert np.all(u == f)

    def test_update_displacement2(self):
        """Test displacement update."""
        nnodes = 3
        u = np.zeros((nnodes, 3))
        f = np.array([
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0]])
        bc_types = np.zeros((nnodes, 3), dtype=np.int32)
        bc_values = np.zeros((nnodes, 3))
        bc_scale = 0
        dt = 2.0
        update_displacement(u, bc_values, bc_types, f, bc_scale, dt)
        assert np.all(u == 2.0*f)

    def test_update_displacement3(self):
        """Test displacement update with displacement boundary conditions."""
        nnodes = 3
        u = np.zeros((nnodes, 3))
        f = np.array([
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0]])
        bc_types = np.array([
            [1, 1, 0],
            [1, 1, 0],
            [1, 1, 0]], dtype=np.int32)
        bc_values = np.zeros((nnodes, 3))
        bc_scale = 1.0
        dt = 2.0
        update_displacement(u, bc_values, bc_types, f, bc_scale, dt)
        u_expected = np.array([[0, 0, 6.0],
                               [0, 0, 6.0],
                               [0, 0, 6.0]])
        assert np.all(u == u_expected)

    def test_update_displacement4(self):
        """Test displacement update with displacement B.C. scale."""
        nnodes = 3
        u = np.zeros((nnodes, 3))
        f = np.array([[1.0, 2.0, 3.0],
                      [1.0, 2.0, 3.0],
                      [1.0, 2.0, 3.0]])
        bc_types = np.array([[1, 1, 0],
                             [1, 1, 0],
                             [1, 1, 0]], dtype=np.int32)
        bc_values = np.array([[2.0, 2.0, 0.0],
                              [2.0, 2.0, 0.0],
                              [2.0, 2.0, 0.0]])
        bc_scale = 0.5
        dt = 2.0
        update_displacement(u, bc_values, bc_types, f, bc_scale, dt)
        u_expected = np.array([[1.0, 1.0, 6.0],
                               [1.0, 1.0, 6.0],
                               [1.0, 1.0, 6.0]])
        assert np.all(u == u_expected)

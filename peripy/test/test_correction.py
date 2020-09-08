"""Tests for the correction module."""
import numpy as np
from peripy.correction import (set_volume_correction,
                               set_imprecise_stiffness_correction,
                               set_precise_stiffness_correction,
                               set_micromodulus_function)


def test_micromodulus_correction():
    """Test the micromodulus correction functions."""
    r0 = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            ])
    nl = np.array([
        [1, 0],
        [0, 2],
        [1, 0]
        ], dtype=np.intc)
    n_neigh = np.array([1, 2, 1], dtype=np.intc)

    mmv_container = np.ones(np.shape(nl), dtype=np.float64)
    actual_mmv = mmv_container.copy()
    horizon = 2.00
    set_micromodulus_function(actual_mmv, r0, nl, n_neigh, horizon, 0)
    expected_mmv = np.array([
        [0.5, 1],
        [0.5, 0.5],
        [0.5, 1]
        ], dtype=np.float64)
    assert np.all(expected_mmv == actual_mmv)
    actual_mmv = mmv_container.copy()
    horizon = 1.00
    set_micromodulus_function(actual_mmv, r0, nl, n_neigh, horizon, 0)
    expected_mmv = np.array([
        [0.0, 1],
        [0.0, 0.0],
        [0.0, 1]
        ], dtype=np.float64)
    assert np.all(expected_mmv == actual_mmv)


def test_volume_correction():
    """Test the volume correction function."""
    node_radius = 0.05
    r0 = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            ])
    nl = np.array([
        [1, 0],
        [0, 2],
        [1, 0]
        ], dtype=np.intc)
    n_neigh = np.array([1, 2, 1], dtype=np.intc)
    vlm_crtn_container = np.ones(np.shape(nl), dtype=np.float64)

    actual_vlm_crtn = vlm_crtn_container.copy()
    horizon = 1.00
    set_volume_correction(actual_vlm_crtn, r0, nl, n_neigh, horizon,
                          node_radius, 0)
    expected_vlm_crtn = np.array([
        [0.5, 1],
        [0.5, 0.5],
        [0.5, 1]
        ], dtype=np.float64)
    assert np.allclose(actual_vlm_crtn, expected_vlm_crtn)

    actual_vlm_crtn = vlm_crtn_container.copy()
    horizon = 1.01
    set_volume_correction(actual_vlm_crtn, r0, nl, n_neigh, horizon,
                          node_radius, 0)
    expected_vlm_crtn = np.array([
        [0.6, 1],
        [0.6, 0.6],
        [0.6, 1]
        ], dtype=np.float64)
    assert np.allclose(actual_vlm_crtn, expected_vlm_crtn)

    actual_vlm_crtn = vlm_crtn_container.copy()
    horizon = 0.95
    set_volume_correction(actual_vlm_crtn, r0, nl, n_neigh, horizon,
                          node_radius, 0)
    expected_vlm_crtn = np.array([
        [0.0, 1],
        [0.0, 0.0],
        [0.0, 1]
        ], dtype=np.float64)
    assert np.allclose(actual_vlm_crtn, expected_vlm_crtn)

    actual_vlm_crtn = vlm_crtn_container.copy()
    horizon = 1.05
    set_volume_correction(actual_vlm_crtn, r0, nl, n_neigh, horizon,
                          node_radius, 0)
    expected_vlm_crtn = np.array([
        [1.0, 1],
        [1.0, 1.0],
        [1.0, 1]
        ], dtype=np.float64)
    expected_vlm_crtn = 1.0
    assert np.allclose(actual_vlm_crtn, expected_vlm_crtn)


class TestStiffnessCorrection():
    """Test the stiffness correction functions."""

    def test_imprecise(self):
        """Test stiffness corrections using average nodal volumes."""
        nl = np.array([
            [1, 0],
            [0, 2],
            [1, 0]
            ], dtype=np.intc)
        n_neigh = np.array([1, 2, 1], dtype=np.intc)
        stf_crtn_container = np.ones(np.shape(nl), dtype=np.float64)
        average_volume = np.float64(1.0)
        family_volume_bulk = np.float64(2.0)
        actual_stf_crtn = stf_crtn_container.copy()
        set_imprecise_stiffness_correction(actual_stf_crtn, nl, n_neigh,
                                           average_volume, family_volume_bulk)
        expected_stf_crtn = np.array([
            [4./3, 1],
            [4./3, 4./3],
            [4./3, 1]
            ], dtype=np.float64)
        assert np.allclose(actual_stf_crtn, expected_stf_crtn)

    def test_precise(self):
        """Test stiffness corrections using precise nodal volumes."""
        volume = np.array([1.0, 2.0, 1.0], dtype=np.float64)
        nl = np.array([
            [1, 0],
            [0, 2],
            [1, 0]
            ], dtype=np.intc)
        n_neigh = np.array([1, 2, 1], dtype=np.intc)
        stf_crtn_container = np.ones(np.shape(nl), dtype=np.float64)
        family_volume_bulk = np.float64(2.0)
        actual_stf_crtn = stf_crtn_container.copy()
        set_precise_stiffness_correction(actual_stf_crtn, nl, n_neigh,
                                         volume, family_volume_bulk)
        expected_stf_crtn = np.array([
            [4./4, 1],
            [4./4, 4./4],
            [4./4, 1]
            ], dtype=np.float64)
        assert np.allclose(actual_stf_crtn, expected_stf_crtn)

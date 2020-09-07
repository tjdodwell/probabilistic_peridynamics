"""Tests for the correction module."""
import numpy as np
from peripy.correction import (cvolume_correction,
                               set_volume_correction,
                               set_imprecise_stiffness_correction,
                               set_precise_stiffness_correction)
from scipy.spatial.distance import euclidean


class TestVolumeCorrection():
    """Test the volume correction function."""
    node_radius = 0.05
    horizon = 1.00
    r10 = np.array([0, 0, 0], dtype=np.float64)
    r20 = np.array([1, 0, 0], dtype=np.float64)
    actual_vlm_crtn = cvolume_correction(r10, r20, horizon, node_radius)
    expected_vlm_crtn = 0.5
    assert actual_vlm_crtn = expected_vlm_crtn
    horizon = 1.01
    actual_vlm_crtn = cvolume_correction(r10, r20, horizon, node_radius)
    expected_vlm_crtn = 0.2
    assert actual_vlm_crtn = expected_vlm_crtn
    horizon = 0.99
    actual_vlm_crtn = cvolume_correction(r10, r20, horizon, node_radius)
    expected_vlm_crtn = 0.0
    assert actual_vlm_crtn = expected_vlm_crtn
    horizon = 1.05
    actual_vlm_crtn = cvolume_correction(r10, r20, horizon, node_radius)
    expected_vlm_crtn = 1.0
    assert actual_vlm_crtn = expected_vlm_crtn

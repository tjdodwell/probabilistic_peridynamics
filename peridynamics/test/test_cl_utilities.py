"""Tests for the cl/utilities module."""
from ..cl import get_context, pad
from ..cl.utilities import DOUBLE_FP_SUPPORT
import numpy as np
import pyopencl as cl


def test_get_context():
    """Test the get_context function."""
    context = get_context()

    if type(context) == cl._cl.Context:
        devices = context.devices
        assert len(devices) == 1
        assert (devices[0].get_info(cl.device_info.DOUBLE_FP_CONFIG)
                & DOUBLE_FP_SUPPORT)
    else:
        assert context is None


class TestPad():
    """Test padding helper function."""

    def test_pad_1d(self):
        """Test padding for a 1D array."""
        dimension = 258
        group_size = 256
        expected_dimension = 512

        array = np.random.random(dimension)
        array = pad(array, group_size)

        assert array.shape == (expected_dimension,)
        assert np.all(
            array[dimension:] == np.zeros(expected_dimension-dimension)
            )

    def test_no_padding(self):
        """Test padding when non is required."""
        dimension = 512
        group_size = 256
        expected_dimension = 512

        array = np.random.random(dimension)
        array = pad(array, group_size)

        assert array.shape == (expected_dimension,)
        assert np.all(
            array[dimension:] == np.zeros(expected_dimension-dimension)
            )

    def test_pad_2d_axis0(self):
        """Test padding a 2D array along axis 0."""
        dimension = 755
        other_dimension = 5
        group_size = 256
        expected_dimension = 768

        array = np.random.random((dimension, other_dimension))
        array = pad(array, group_size)

        assert array.shape == ((expected_dimension, other_dimension,))
        assert np.all(
            array[dimension:, :] ==
            np.zeros((expected_dimension-dimension, other_dimension))
            )

    def test_pad_2d_axis1(self):
        """Test padding a 2D array along axis 1."""
        dimension = 400
        other_dimension = 17
        group_size = 256
        expected_dimension = 512

        array = np.random.random((other_dimension, dimension))
        array = pad(array, group_size, axis=1)

        assert array.shape == ((other_dimension, expected_dimension, ))
        assert np.all(
            array[:, dimension:] ==
            np.zeros((other_dimension, expected_dimension-dimension))
            )

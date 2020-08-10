"""Tests for the cl/utilities module."""
from ..cl import get_context
from ..cl.utilities import DOUBLE_FP_SUPPORT, output_device_info
import pyopencl as cl


def test_get_context():
    """Test the get_context function."""
    context = get_context()

    if type(context) == cl._cl.Context:
        devices = context.devices
        assert len(devices) == 1
        assert (devices[0].get_info(cl.device_info.DOUBLE_FP_CONFIG)
                & DOUBLE_FP_SUPPORT)
        assert (output_device_info(devices[0]) == 1)
    else:
        assert context is None

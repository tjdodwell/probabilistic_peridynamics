"""Utilities for using the OpenCL kernels."""
import numpy as np
import pyopencl as cl


DOUBLE_FP_SUPPORT = (
    cl.device_fp_config.DENORM | cl.device_fp_config.FMA |
    cl.device_fp_config.INF_NAN | cl.device_fp_config.ROUND_TO_INF |
    cl.device_fp_config.ROUND_TO_NEAREST |
    cl.device_fp_config.ROUND_TO_ZERO
    )


def get_context():
    """
    Find an appropriate OpenCL context.

    This function looks for a device with support for double
    floating-point precision and prefers GPU devices.

    :returns: A context with a single suitable device, or `None` is no suitable
    device is found.
    :rtype: :class:`pyopencl._cl.Context` or `NoneType`
    """
    for platform in cl.get_platforms():
        for device_type in [cl.device_type.GPU, cl.device_type.ALL]:
            for device in platform.get_devices(device_type):
                if (device.get_info(cl.device_info.DOUBLE_FP_CONFIG)
                        == DOUBLE_FP_SUPPORT):
                    return cl.Context([device])
    return None


def pad(array, group_size, axis=0):
    """
    Pad an array with zeros so that it is a multiple of the group size.

    :arg array: Array to pad.
    :type array: :class:`numpy.ndarray`
    :arg int group_size: OpenCL group size.
    :arg int axis: The axis to pad with zeros. Default is 0.

    :returns: `array` padded with an appropriate number of zeros.
    :rtype: :class:`numpy.ndarray`
    """
    array_size = array.shape[axis]
    remainder = array_size % group_size
    if remainder == 0:
        return array
    else:
        padding = group_size - array_size % group_size
        padding_shape = list(array.shape)
        padding_shape[axis] = padding
        return np.concatenate(
            (array, np.zeros(padding_shape, dtype=array.dtype)), axis=axis
            )

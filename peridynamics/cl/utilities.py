"""Utilities for using teh OpenCL kernels."""
import numpy as np


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
        return np.concatenate((array, np.zeros(padding_shape)), axis=axis)

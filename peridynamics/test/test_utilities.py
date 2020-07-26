"""Tests for the utilities module."""
import numpy as np
from peridynamics.utilities import (read_array, _calc_midpoint_gradient,
                                    calc_displacement_scale, calc_build_time)
import pytest
import pathlib


def test_read_array():
    write_path = pathlib.path
    expected_array = np.empty((2113, 256)
    write_array(expected_array, "name")
    read_array(write_path, "name")
    assert np.all(array == expected_array)

# test that the midpoint gradient of a known 5th order polynomial is calculated
# correctly.
# test that the resulting polynomial has 0 gradient at start and finish

def test_calc_midpoint_gradient():
    """
    Calculate the midpoint gradient and coefficients of a 5th order polynomial.

    Calculates the midpoint gradient and coefficients of a 5th order
    polynomial displacement-time curve which is defined by acceleration being
    0 at t=0 and t=T and a midpoint gradient.

    :arg int T: The finish time-step of the displacement-time curve.
    :arg float displacement_scale_rate: The midpoint gradient of the curve.

    :returns: A tuple containing a float
        the midpoint gradient of the displacement-time curve and a tuple
        containing the 3 unconstrained coefficients of the 5th-order
        polynomial.
    :rtype: A tuple containing (:type float:, :type tuple:)
    """
    pass


def test_calc_displacement_scale():
    pass


def test_calc_build_time():
    pass

"""Tests for the utilities module."""
import numpy as np
from peridynamics.utilities import (read_array, write_array,
                                    _calc_midpoint_gradient,
                                    calc_build_time)
import pytest
import os


def test_read_and_write_array(data_path):
    rw_path = data_path/"test_rw_array.h5"
    expected_array = np.empty((2113, 256))
    write_array(rw_path, "name", expected_array)
    array = read_array(rw_path, "name")
    assert np.all(array == expected_array)
    with pytest.warns(UserWarning) as warning:
        read_array(rw_path, "no_name")
        assert "array does not appear to exist" in str(warning[0].message)
        os.remove(rw_path)


def test_read_bad_file(data_path):
    read_path = data_path/"no_file_here.h5"
    with pytest.warns(UserWarning) as warning:
        read_array(read_path, "name")
        assert "file does not appear to exist" in str(warning[0].message)


def test_calc_midpoint_gradient():
    """Test calculation of 5th order smooth polynomial."""
    T = 1000
    expected_displacement = 0.005
    expected_velocity = 0.0
    expected_acceleration = 0.0
    (midpoint_gradient, coefficients) = _calc_midpoint_gradient(
        T, expected_displacement)
    a, b, c = coefficients
    displacement = a * T**5 + b * T**4 + c * T**3
    velocity = 5 * a * T**4 + 4 * b * T**3 + 3 * c * T**2
    acceleration = 20 * a * T**3 + 12 * b * T**2 + 6 * c * T
    assert np.isclose(expected_displacement, displacement)
    assert np.isclose(expected_velocity, velocity)
    assert np.isclose(expected_acceleration, acceleration)


def test_calc_build_time():
    """ Test calculation of the build-up time steps."""
    steps = 1000
    expected_build_displacement = 0.0025
    expected_velocity = 0.0
    expected_acceleration = 0.0
    expected_max_velocity = 0.00001
    build_time, (a, b, c) = calc_build_time(
        expected_build_displacement, expected_max_velocity, steps)
    midpoint_time = build_time/2
    actual_build_displacement = (
        a * build_time**5 + b * build_time**4 + c * build_time**3)
    actual_velocity = (
        5 * a * build_time**4 + 4 * b * build_time**3 + 3 * c * build_time**2)
    actual_acceleration = (
        20 * a * build_time**3 + 12 * b * build_time**2 + 6 * c * build_time)
    actual_max_velocity = (
        5 * a * midpoint_time**4
        + 4 * b * midpoint_time**3
        + 3 * c * midpoint_time**2)

    assert np.isclose(actual_build_displacement, expected_build_displacement)
    assert np.isclose(actual_velocity, expected_velocity)
    assert np.isclose(actual_acceleration, expected_acceleration)
    assert np.isclose(actual_max_velocity, expected_max_velocity)

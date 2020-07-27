"""Tests for the utilities module."""
import numpy as np
from peridynamics.utilities import (read_array, write_array,
                                    _calc_midpoint_gradient,
                                    calc_displacement_scale, calc_build_time)
import pytest
import pathlib


def test_read_and_write_array():
    write_path = pathlib.path
    expected_array = np.empty((2113, 256))
    write_array(expected_array, "name")
    array = read_array(write_path, "name")
    assert np.all(array == expected_array)


def test_read_bad_file():
    write_path = pathlib.path
    expected_array = np.empty((2113, 256))
    write_array(expected_array, "name")
    with pytest.raises(IOError) as exception:
        read_array(write_path, "name")
        assert ("The .h5 file") in exception.value


def test_read_bad_array():
    write_path = pathlib.path
    expected_array = np.empty((2113, 256))
    write_array(expected_array, "name")
    with pytest.raises(IOError) as exception:
        read_array(write_path, "no_name")
        assert ("The array") in exception.value


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


def test_calc_displacement_scale():
    """Test calculation of displacement boundary condition scale."""

    steps = 1000
    ease_off = 0
    build_displacement = 0.0025
    max_displacement_rate = 0.00001
    max_displacement = 0.005
    build_time, coefficients = calc_build_time(
        build_displacement, max_displacement_rate, steps)

    step = int(build_time/2 - 1)
    displacement_bc_rate, ease_off = calc_displacement_scale(
        coefficients, max_displacement, build_time, max_displacement_rate,
        step, build_displacement, ease_off)
    assert displacement_bc_rate < max_displacement_rate
    assert ease_off == 0

    step = int(build_time/2 + 1)
    displacement_bc_rate, ease_off = calc_displacement_scale(
        coefficients, max_displacement, build_time, max_displacement_rate,
        step, build_displacement, ease_off)
    assert displacement_bc_rate == max_displacement_rate
    assert ease_off == 0

    ease_off = 990 - build_time/2
    step = 991
    displacement_bc_rate, ease_off = calc_displacement_scale(
        coefficients, max_displacement, build_time, max_displacement_rate,
        step, build_displacement, ease_off)
    assert displacement_bc_rate == 0
    assert ease_off > build_time/2

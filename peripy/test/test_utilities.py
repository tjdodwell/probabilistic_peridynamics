"""Tests for the utilities module."""
import numpy as np
from peripy.utilities import (read_array, write_array)
import pytest


def test_read_and_write_array(tmpdir):
    """Test reading and writing an array."""
    rw_path = tmpdir/"test_rw_array.h5"
    expected_array = np.ones((2113, 256))
    write_array(rw_path, "name", expected_array)
    array = read_array(rw_path, "name")
    assert np.all(array == expected_array)
    with pytest.warns(UserWarning) as warning:
        read_array(rw_path, "no_name")
        assert "array does not appear to exist" in str(warning[0].message)


def test_read_bad_file(data_path):
    """Test reading a file that doesn't exist."""
    read_path = data_path/"no_file_here.h5"
    with pytest.warns(UserWarning) as warning:
        read_array(read_path, "name")
        assert "file does not appear to exist" in str(warning[0].message)

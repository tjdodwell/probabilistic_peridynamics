"""Tests for the OpenCL kernels."""
from .conftest import context_available
from ..cl import get_context, kernel_source
import numpy as np
from peridynamics.neighbour_list import (create_neighbour_list_cl, set_family)
import pyopencl as cl
from pyopencl import mem_flags as mf
import pytest


@pytest.fixture(scope="module")
def context():
    """Create a context using the default platform, prefer GPU."""
    return get_context()


@context_available
@pytest.fixture(scope="module")
def queue(context):
    """Create a CL command queue."""
    return cl.CommandQueue(context)


@context_available
@pytest.fixture(scope="module")
def program(context):
    """Create a program object from the kernel source."""
    return cl.Program(context, kernel_source).build()


class TestUpdateDisplacement():
    """Test displacement update calculation."""

    def test_update_displacement():
        pass

"""Tests for the OpenCL kernels."""
from .conftest import context_available
from ..cl import get_context
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
    """Create a program object for the Euler integrator."""
    kernel_source = open(
            pathlib.Path(__file__).parent.absolute() /
            "../cl/euler.cl").read()
    return cl.Program(context, kernel_source).build()


class TestUpdateDisplacement:
    """Test the displacement update."""

    def test_update_displacement(self, context, queue, program):
        """Test basic displacement update."""
        u = np.zeros(3)
        nnodes = 1
        force = np.array([1.0, 2.0, 3.0])
        bc_types = np.array([0, 0, 0])
        bc_values = np.array([0, 0, 0])
        displacement_load_scale = 0
        dt = 1
        update_displacement(u, bc_values, bc_types, f, bc_scale, dt)

        # Set buffers
        # Read only
        r0_d = cl.Buffer(
            context, mf.READ_ONLY | mf.COPY_HOST_PTR,
            hostbuf=coords)
        vols_d = cl.Buffer(
            context, mf.READ_ONLY | mf.COPY_HOST_PTR,
            hostbuf=volume)
        family_d = cl.Buffer(
            context, mf.READ_ONLY | mf.COPY_HOST_PTR,
            hostbuf=family)
        bc_types_d = cl.Buffer(
            context, mf.READ_ONLY | mf.COPY_HOST_PTR,
            hostbuf=bc_types)
        bc_values_d = cl.Buffer(
            context, mf.READ_ONLY | mf.COPY_HOST_PTR,
            hostbuf=bc_values)
        force_d = cl.Buffer(
            context, mf.READ_WRITE, force.nbytes)
        u_d = cl.Buffer(
            context, mf.READ_WRITE | mf.COPY_HOST_PTR,
            hostbuf=u)
        # Write only
        damage_d = cl.Buffer(
            context, mf.WRITE_ONLY, damage.nbytes)

        # Build kernels
        update_displacement_kernel = program.update_displacement

        update_displacement_kernel(
            queue, (3 * nnodes,), None,
            force_d, u_d, bc_types_d, bc_values_d,
            np.float64(displacement_load_scale), np.float64(dt))
        assert np.all(u == f)

    def test_update_displacement2(self):
        """Test displacement update."""
        u = np.zeros(3)
        f = np.array([1.0, 2.0, 3.0])
        bc_types = np.array([0, 0, 0])
        bc_values = np.array([0, 0, 0])
        bc_scale = 0
        dt = 2.0
        update_displacement(u, bc_values, bc_types, f, bc_scale, dt)
        assert np.all(u == 2.0*f)

    def test_update_displacement3(self):
        """Test displacement update with displacement boundary conditions."""
        u = np.zeros(3)
        f = np.array([1.0, 2.0, 3.0])
        bc_types = np.array([1, 1, 0])
        bc_values = np.array([0.0, 0.0, 0.0])
        bc_scale = 1.0
        dt = 2.0
        update_displacement(u, bc_values, bc_types, f, bc_scale, dt)
        u_expected = np.array([0, 0, 6.0])
        assert np.all(u == u_expected)

    def test_update_displacement4(self):
        """Test displacement update with displacement B.C. scale."""
        u = np.zeros(3)
        f = np.array([1.0, 2.0, 3.0])
        bc_types = np.array([1, 1, 0])
        bc_values = np.array([2.0, 2.0, 0.0])
        bc_scale = 0.5
        dt = 2.0
        update_displacement(u, bc_values, bc_types, f, bc_scale, dt)
        u_expected = np.array([1.0, 1.0, 6.0])
        assert np.all(u == u_expected)

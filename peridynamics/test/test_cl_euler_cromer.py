"""Tests for the OpenCL kernels."""
from .conftest import context_available
from ..cl import get_context
import numpy as np
import pyopencl as cl
from pyopencl import mem_flags as mf
import pathlib
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
            "../cl/euler_cromer.cl").read()
    return cl.Program(context, kernel_source).build()


class TestUpdateDisplacement:
    """Test the displacement update."""

    @context_available
    def test_update_displacement(self, context, queue, program):
        """Test basic displacement update."""
        u = np.zeros(3)
        ud = np.zeros(3)
        densities = np.ones(3)
        nnodes = 1
        force = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        bc_types = np.array([0, 0, 0], dtype=np.intc)
        bc_values = np.array([0, 0, 0], dtype=np.float64)
        displacement_bc_scale = 0
        dt = 1
        damping = 1

        # Set buffers
        # Read only
        bc_types_d = cl.Buffer(
            context, mf.READ_ONLY | mf.COPY_HOST_PTR,
            hostbuf=bc_types)
        bc_values_d = cl.Buffer(
            context, mf.READ_ONLY | mf.COPY_HOST_PTR,
            hostbuf=bc_values)
        force_d = cl.Buffer(
            context, mf.READ_ONLY | mf.COPY_HOST_PTR,
            hostbuf=force)
        # Read write
        u_d = cl.Buffer(
            context, mf.READ_WRITE | mf.COPY_HOST_PTR,
            hostbuf=u)
        ud_d = cl.Buffer(
            context, mf.READ_WRITE | mf.COPY_HOST_PTR,
            hostbuf=ud)
        densities_d = cl.Buffer(
            context, mf.READ_WRITE | mf.COPY_HOST_PTR,
            hostbuf=densities)

        # Build kernels
        update_displacement_kernel = program.update_displacement

        update_displacement_kernel(
            queue, (3 * nnodes,), None,
            force_d, u_d, ud_d, bc_types_d, bc_values_d, densities_d,
            np.float64(displacement_bc_scale), np.float64(dt),
            np.float64(damping))
        cl.enqueue_copy(queue, u, u_d)
        cl.enqueue_copy(queue, ud, ud_d)

        assert np.all(u == force)
        assert np.all(ud == force)

    @context_available
    def test_update_displacement2(self, context, queue, program):
        """Test displacement update."""
        u = np.zeros(3)
        ud = np.zeros(3)
        densities = np.ones(3)
        nnodes = 1
        force = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        bc_types = np.array([0, 0, 0], dtype=np.intc)
        bc_values = np.array([0, 0, 0], dtype=np.float64)
        displacement_bc_scale = 0
        dt = 2.0
        damping = 1.0

        # Set buffers
        # Read only
        bc_types_d = cl.Buffer(
            context, mf.READ_ONLY | mf.COPY_HOST_PTR,
            hostbuf=bc_types)
        bc_values_d = cl.Buffer(
            context, mf.READ_ONLY | mf.COPY_HOST_PTR,
            hostbuf=bc_values)
        force_d = cl.Buffer(
            context, mf.READ_ONLY | mf.COPY_HOST_PTR,
            hostbuf=force)
        # Read write
        u_d = cl.Buffer(
            context, mf.READ_WRITE | mf.COPY_HOST_PTR,
            hostbuf=u)
        ud_d = cl.Buffer(
            context, mf.READ_WRITE | mf.COPY_HOST_PTR,
            hostbuf=ud)
        densities_d = cl.Buffer(
            context, mf.READ_WRITE | mf.COPY_HOST_PTR,
            hostbuf=densities)

        # Build kernels
        update_displacement_kernel = program.update_displacement

        update_displacement_kernel(
            queue, (3 * nnodes,), None,
            force_d, u_d, ud_d, bc_types_d, bc_values_d, densities_d,
            np.float64(displacement_bc_scale), np.float64(dt),
            np.float64(damping))
        cl.enqueue_copy(queue, u, u_d)
        cl.enqueue_copy(queue, ud, ud_d)

        assert np.all(u == 4.0*force)
        assert np.all(ud == 2.0*force)

    @context_available
    def test_update_displacement3(self, context, queue, program):
        """Test displacement update with displacement boundary conditions."""
        u = np.zeros(3, dtype=np.float64)
        ud = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        densities = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        nnodes = 1
        force = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        bc_types = np.array([1, 1, 0], dtype=np.intc)
        bc_values = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        displacement_bc_scale = 1.0
        dt = 2.0
        damping = 2.0

        # Set buffers
        # Read only
        bc_types_d = cl.Buffer(
            context, mf.READ_ONLY | mf.COPY_HOST_PTR,
            hostbuf=bc_types)
        bc_values_d = cl.Buffer(
            context, mf.READ_ONLY | mf.COPY_HOST_PTR,
            hostbuf=bc_values)
        force_d = cl.Buffer(
            context, mf.READ_ONLY | mf.COPY_HOST_PTR,
            hostbuf=force)
        # Read write
        u_d = cl.Buffer(
            context, mf.READ_WRITE | mf.COPY_HOST_PTR,
            hostbuf=u)
        ud_d = cl.Buffer(
            context, mf.READ_WRITE | mf.COPY_HOST_PTR,
            hostbuf=ud)
        densities_d = cl.Buffer(
            context, mf.READ_WRITE | mf.COPY_HOST_PTR,
            hostbuf=densities)

        # Build kernels
        update_displacement_kernel = program.update_displacement

        update_displacement_kernel(
            queue, (3 * nnodes,), None,
            force_d, u_d, ud_d, bc_types_d, bc_values_d, densities_d,
            np.float64(displacement_bc_scale), np.float64(dt),
            np.float64(damping))
        cl.enqueue_copy(queue, u, u_d)
        cl.enqueue_copy(queue, ud, ud_d)
        u_expected = np.array([0.0, 0.0, 6.0])
        ud_expected = np.array([-1.0, 1.0, 3.0])

        assert np.all(u == u_expected)
        assert np.all(ud == ud_expected)

    @context_available
    def test_update_displacement4(self, context, queue, program):
        """Test displacement update with displacement B.C. scale."""
        u = np.zeros(3)
        ud = np.zeros(3)
        densties = np.ones(3)
        nnodes = 1
        force = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        bc_types = np.array([1, 1, 0], dtype=np.intc)
        bc_values = np.array([2.0, 2.0, 0.0], dtype=np.float64)
        displacement_bc_scale = 0.5
        dt = 2.0
        damping = 1.0

        # Set buffers
        # Read only
        bc_types_d = cl.Buffer(
            context, mf.READ_ONLY | mf.COPY_HOST_PTR,
            hostbuf=bc_types)
        bc_values_d = cl.Buffer(
            context, mf.READ_ONLY | mf.COPY_HOST_PTR,
            hostbuf=bc_values)
        force_d = cl.Buffer(
            context, mf.READ_ONLY | mf.COPY_HOST_PTR,
            hostbuf=force)
        # Read write
        u_d = cl.Buffer(
            context, mf.READ_WRITE | mf.COPY_HOST_PTR,
            hostbuf=u)
        ud_d = cl.Buffer(
            context, mf.READ_WRITE | mf.COPY_HOST_PTR,
            hostbuf=ud)
        densities_d = cl.Buffer(
            context, mf.READ_WRITE | mf.COPY_HOST_PTR,
            hostbuf=densties)

        # Build kernels
        update_displacement_kernel = program.update_displacement

        update_displacement_kernel(
            queue, (3 * nnodes,), None,
            force_d, u_d, ud_d, bc_types_d, bc_values_d, densities_d,
            np.float64(displacement_bc_scale), np.float64(dt),
            np.float64(damping))
        cl.enqueue_copy(queue, u, u_d)
        cl.enqueue_copy(queue, ud, ud_d)
        u_expected = np.array([1.0, 1.0, 12.0])
        ud_expected = np.array([2.0, 4.0, 6.0])

        assert np.all(u == u_expected)
        assert np.all(ud == ud_expected)

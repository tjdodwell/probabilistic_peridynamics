"""Tests for the OpenCL kernels."""
from ..cl import kernel_source
import numpy as np
import pyopencl as cl
from pyopencl import mem_flags as mf
import pytest
from scipy.spatial.distance import cdist


@pytest.fixture(scope="module")
def gpu_context():
    """Create a context with the first GPU on the default platform."""
    platform = cl.get_platforms()
    devices = platform[0].get_devices(cl.device_type.GPU)
    return cl.Context([devices[0]])


@pytest.fixture(scope="module")
def queue(gpu_context):
    """Create a CL command queue."""
    return cl.CommandQueue(gpu_context)


@pytest.fixture(scope="module")
def program(gpu_context):
    """Create a program object from the kernel source."""
    return cl.Program(gpu_context, kernel_source).build()


def test_distance(gpu_context, queue, program):
    """Test Euclidean distance calculation."""
    # Create coordinates
    n = 1000
    r = np.random.random((n, 3)).astype(np.float32)
    d = np.empty((n, n), dtype=np.float32)

    # Kernel functor
    dist = program.dist

    # Create buffers
    r_d = cl.Buffer(gpu_context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=r)
    d_d = cl.Buffer(gpu_context, mf.WRITE_ONLY, d.nbytes)

    dist(queue, (n, n), None, r_d, d_d)
    cl.enqueue_copy(queue, d, d_d)
    assert np.allclose(d, cdist(r, r))


def test_strain(gpu_context, queue, program):
    """Test strain calculation."""
    def _strain(r, d0):
        d = cdist(r, r)
        strain = (d - d0) / (d0 + np.eye(r.shape[0]))
        return strain

    # Create test data
    n = 1000
    r = np.random.random((n, 3)).astype(np.float32)
    u = np.random.random((n, 3)).astype(np.float32)
    d0 = cdist(r, r).astype(np.float32)
    strain_h = np.empty_like(d0)

    # Kernel functor
    strain = program.strain

    # Create buffers
    r_d = cl.Buffer(gpu_context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=r+u)
    d0_d = cl.Buffer(gpu_context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=d0)
    strain_d = cl.Buffer(gpu_context, mf.WRITE_ONLY, strain_h.nbytes)

    strain(queue, (n, n), None, r_d, d0_d, strain_d)
    cl.enqueue_copy(queue, strain_h, strain_d)
    assert np.allclose(strain_h, _strain(r+u, d0), atol=1.e-6)

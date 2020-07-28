"""Tests for the OpenCL kernels."""
from .conftest import context_available
from ..cl import get_context, kernel_source
import numpy as np
from peridynamics.neighbour_list import (create_neighbour_list_cl)
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


class TestForce():
    """Test force calculation."""

    @context_available
    def test_initial_force(self, context, queue, program):
        """Ensure forces are zero when there is no displacement."""
        r0 = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            ], dtype=np.float64)

        horizon = 1.1
        nnodes = 5
        u = np.zeros((nnodes, 3), dtype=np.float64)
        volume = np.ones(nnodes, dtype=np.float64)
        bond_stiffness = 1.0
        critical_stretch = 1.0
        max_neigh = 4
        nlist, n_neigh = create_neighbour_list_cl(r0, horizon, max_neigh)

        force_load_scale = 1.0
        force_bc_types = np.zeros((nnodes, 3), dtype=np.float64)
        force_bc_values = np.zeros((nnodes, 3), dtype=np.float64)

        force_expected = np.zeros((nnodes, 3), dtype=np.float64)
        force_actual = np.empty_like(force_expected)

        # Create buffers
        # Read and write
        nlist_d = cl.Buffer(
            context, mf.READ_WRITE | mf.COPY_HOST_PTR,
            hostbuf=nlist)
        n_neigh_d = cl.Buffer(
            context, mf.READ_WRITE | mf.COPY_HOST_PTR,
            hostbuf=n_neigh)
        local_mem_x = cl.LocalMemory(
            np.dtype(np.float64).itemsize * max_neigh)
        local_mem_y = cl.LocalMemory(
            np.dtype(np.float64).itemsize * max_neigh)
        local_mem_z = cl.LocalMemory(
            np.dtype(np.float64).itemsize * max_neigh)
        # Read only
        u_d = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR,
                        hostbuf=u)
        r0_d = cl.Buffer(
            context, mf.READ_ONLY | mf.COPY_HOST_PTR,
            hostbuf=r0)
        nlist_d = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR,
                            hostbuf=nlist)
        n_neigh_d = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR,
                              hostbuf=n_neigh)
        vols_d = cl.Buffer(
            context, mf.READ_ONLY | mf.COPY_HOST_PTR,
            hostbuf=volume)
        force_bc_types_d = cl.Buffer(
           context, mf.READ_ONLY | mf.COPY_HOST_PTR,
           hostbuf=force_bc_types)
        force_bc_values_d = cl.Buffer(
            context, mf.READ_ONLY | mf.COPY_HOST_PTR,
            hostbuf=force_bc_values)
        # Write only
        force_d = cl.Buffer(context, mf.WRITE_ONLY, force_expected.nbytes)

        # Call kernel
        bond_force = program.bond_force
        bond_force(
            queue, (nnodes * max_neigh,),
            (max_neigh,), u_d, force_d, r0_d, vols_d, nlist_d,
            n_neigh_d, force_bc_types_d, force_bc_values_d, local_mem_x,
            local_mem_y, local_mem_z, np.float64(force_load_scale),
            np.float64(bond_stiffness), np.float64(critical_stretch))

        cl.enqueue_copy(queue, force_actual, force_d)

        assert np.allclose(force_actual, force_expected)

    @context_available
    def test_force(self, context, queue, program):
        """Ensure forces are in the correct direction using a minimal model."""
        r0 = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            ], dtype=np.float64)
        horizon = 1.01
        elastic_modulus = 0.05
        bond_stiffness = 18.0 * elastic_modulus / (np.pi * horizon**4)
        max_neigh = 3
        volume = np.full(3, 0.16666667, dtype=np.float64)
        nlist, n_neigh = create_neighbour_list_cl(r0, horizon, max_neigh)

        # Displace particles, but do not update neighbour list
        r = r0 + np.array([
            [0.0, 0.0, 0.0],
            [0.05, 0.0, 0.0],
            [0.05, 0.05, 0.0]
            ], dtype=np.float64)

        force_value = 0.00229417
        force_expected = np.array([
            [force_value, 0., 0.],
            [-force_value, force_value, 0.],
            [0., -force_value, 0.]
            ])
        force_actual = np.empty_like(force_expected)

        # Create buffers
        r_d = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR,
                        hostbuf=r)
        r0_d = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR,
                         hostbuf=r0)
        nlist_d = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR,
                            hostbuf=nlist)
        n_neigh_d = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR,
                              hostbuf=n_neigh)
        volume_d = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR,
                             hostbuf=volume)
        force_d = cl.Buffer(context, mf.WRITE_ONLY, force_expected.nbytes)

        # Call kernel
        bond_force = program.bond_force
        bond_force(queue, n_neigh.shape, None, r_d, r0_d, nlist_d, n_neigh_d,
                   np.int32(max_neigh), volume_d, np.float64(bond_stiffness),
                   force_d)
        cl.enqueue_copy(queue, force_actual, force_d)

        assert np.allclose(force_actual, force_expected)


@context_available
def test_break_bonds(context, queue, program):
    """Test neighbour list function."""
    r0 = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [2.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        ])
    horizon = 1.1
    max_neigh = 3
    nl, n_neigh = create_neighbour_list_cl(r0, horizon, max_neigh)

    nl_expected = np.array([
        [1, 2, 4],
        [0, 3, 0],
        [0, 0, 0],
        [1, 0, 0],
        [0, 0, 0]
        ])
    n_neigh_expected = np.array([3, 2, 1, 1, 1])

    assert np.all(nl == nl_expected)
    assert np.all(n_neigh == n_neigh_expected)

    r = np.array([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [3.0, 0.0, 0.0],
        [0.0, 0.0, 2.0],
        ])
    critical_strain = 1.0

    # Create buffers
    r_d = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=r)
    r0_d = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=r0)
    nlist_d = cl.Buffer(context, mf.READ_WRITE | mf.COPY_HOST_PTR,
                        hostbuf=nl)
    n_neigh_d = cl.Buffer(context, mf.READ_WRITE | mf.COPY_HOST_PTR,
                          hostbuf=n_neigh)

    # Call kernel
    break_bonds = program.break_bonds
    break_bonds(queue, n_neigh.shape, None, r_d, r0_d, nlist_d, n_neigh_d,
                np.int32(max_neigh), np.float64(critical_strain))
    cl.enqueue_copy(queue, nl, nlist_d)
    cl.enqueue_copy(queue, n_neigh, n_neigh_d)

    nl_expected = np.array([
        [2, 2, 4],
        [3, 3, 0],
        [0, 0, 0],
        [1, 0, 0],
        [0, 0, 0]
        ])
    n_neigh_expected = np.array([1, 1, 1, 1, 0])

    assert np.all(nl == nl_expected)
    assert np.all(n_neigh == n_neigh_expected)

"""Peridynamics model using OpenCL kernels."""
from .model import Model
from .cl import double_fp_support, get_context, kernel_source
import numpy as np
import pyopencl as cl
from pyopencl import mem_flags as mf


class ModelCL(Model):
    """OpenCL Model."""

    def __init__(self, *args, context=None, **kwargs):
        """Create a :class:`ModelCL` object."""
        super().__init__(*args, **kwargs)

        # Get an OpenCL context if none was provided
        if context is None:
            self.context = get_context()
            # Ensure that self.context is a pyopencl context object
            if type(self.context) is not cl._cl.Context:
                raise ContextError
        else:
            self.context = context
            # Ensure that self.context is a pyopencl context object
            if type(self.context) is not cl._cl.Context:
                raise TypeError("context must be a pyopencl Context object")
            # Ensure that self.context supports double floating-point precssion
            if not double_fp_support(self.context.devices[0]):
                raise ValueError("device 0 of context must support double"
                                 "floating-point precision")

        # Build kernels
        self.program = cl.Program(self.context, kernel_source).build()
        self.queue = cl.CommandQueue(self.context)

        self.damage_kernel = self.program.damage
        self.break_bonds_kernel = self.program.break_bonds
        self.bond_force_kernel = self.program.bond_force

    def _damage(self, n_neigh):
        """
        Calculate bond damage.

        :arg n_neigh: The number of neighbours of each node.
        :type n_neigh: :class:`numpy.ndarray`

        :returns: A (`nnodes`, ) array containing the damage for each node.
        :rtype: :class:`numpy.ndarray`
        """
        context = self.context
        queue = self.queue

        damage = np.empty(n_neigh.shape, dtype=np.float64)

        # Create buffers
        n_neigh_d = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR,
                              hostbuf=n_neigh)
        family_d = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR,
                             hostbuf=self.family)
        damage_d = cl.Buffer(context, mf.WRITE_ONLY, damage.nbytes)

        # Call kernel
        self.damage_kernel(queue, damage.shape, None, n_neigh_d, family_d,
                           damage_d)
        queue.finish()
        cl.enqueue_copy(queue, damage, damage_d)
        return damage

    def _break_bonds(self, u, nlist, n_neigh):
        """
        Break bonds which have exceeded the critical strain.

        :arg u: A (nnodes, 3) array of the displacements of each node.
        :type u: :class:`numpy.ndarray`
        :arg nlist: The neighbour list.
        :type nlist: :class:`numpy.ndarray`
        :arg n_neigh: The number of neighbours of each node.
        :type n_neigh: :class:`numpy.ndarray`
        """
        context = self.context
        queue = self.queue

        # Create buffers
        r_d = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR,
                        hostbuf=self.coords+u)
        r0_d = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR,
                         hostbuf=self.coords)
        nlist_d = cl.Buffer(context, mf.READ_WRITE | mf.COPY_HOST_PTR,
                            hostbuf=nlist)
        n_neigh_d = cl.Buffer(context, mf.READ_WRITE | mf.COPY_HOST_PTR,
                              hostbuf=n_neigh)

        # Call kernel
        self.break_bonds_kernel(queue, n_neigh.shape, None, r_d, r0_d, nlist_d,
                                n_neigh_d, np.int32(self.max_neighbours),
                                np.float64(self.critical_strain))
        queue.finish()
        cl.enqueue_copy(queue, nlist, nlist_d)
        cl.enqueue_copy(queue, n_neigh, n_neigh_d)

    def _bond_force(self, u, nlist, n_neigh):
        """
        Calculate the force due to bonds acting on each node.

        :arg u: A (nnodes, 3) array of the displacements of each node.
        :type u: :class:`numpy.ndarray`
        :arg nlist: The neighbour list.
        :type nlist: :class:`numpy.ndarray`
        :arg n_neigh: The number of neighbours of each node.
        :type n_neigh: :class:`numpy.ndarray`

        :returns: A (`nnodes`, 3) array of the component of the force in each
            dimension for each node.
        :rtype: :class:`numpy.ndarray`
        """
        context = self.context
        queue = self.queue

        force = np.empty_like(self.coords)

        # Create buffers
        r_d = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR,
                        hostbuf=self.coords+u)
        r0_d = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR,
                         hostbuf=self.coords)
        nlist_d = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR,
                            hostbuf=nlist)
        n_neigh_d = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR,
                              hostbuf=n_neigh)
        volume_d = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR,
                             hostbuf=self.volume)
        force_d = cl.Buffer(context, mf.WRITE_ONLY, force.nbytes)

        # Call kernel
        self.bond_force_kernel(queue, n_neigh.shape, None, r_d, r0_d, nlist_d,
                               n_neigh_d, np.int32(self.max_neighbours),
                               volume_d, np.float64(self.bond_stiffness),
                               force_d)
        queue.finish()
        cl.enqueue_copy(queue, force, force_d)
        return force


class ContextError(Exception):
    """No suitable context was found by :func:`get_context`."""

    def __init__(self):
        """Exception constructor."""
        message = ("No suitable context was found. You can manually specify"
                   "the context by passing it to ModelCL with the 'context'"
                   "argument.")
        super().__init__(message)

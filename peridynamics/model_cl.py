"""Peridynamics model using OpenCL kernels."""
from .model import Model
from .cl import get_context, kernel_source
import numpy as np
import pyopencl as cl
from pyopencl import mem_flags as mf


class ModelCL(Model):
    """OpenCL Model."""

    def __init__(self, *args, context=None, **kwargs):
        super().__init__(*args, **kwargs)

        # Get an OpenCL context
        if context is None:
            self.context = get_context()
        else:
            self.context = context
        assert type(self.context) is cl._cl.Context

        # Build kernels
        self.program = cl.Program(self.context, kernel_source).build()
        self.queue = cl.CommandQueue(self.context)

        self.damage_kernel = self.program.damage
        self.bond_force_kernel = self.program.bond_force

    def _damage(self, n_neigh):
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
        cl.enqueue_copy(queue, damage, damage_d)
        return damage

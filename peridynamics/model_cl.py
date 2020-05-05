"""Peridynamics model using OpenCL kernels."""
from .integrators import Integrator
from .model import Model, InvalidIntegrator
from .cl import double_fp_support, get_context, kernel_source
import numpy as np
import pathlib
import pyopencl as cl
from pyopencl import mem_flags as mf
from tqdm import trange


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

    def simulate(self, steps, integrator, boundary_function=None, u=None,
                 connectivity=None, first_step=1, write=None, write_path=None):
        """
        Simulate the peridynamics model.

        :arg int steps: The number of simulation steps to conduct.
        :arg  integrator: The integrator to use, see
            :mod:`peridynamics.integrators` for options.
        :type integrator: :class:`peridynamics.integrators.Integrator`
        :arg boundary_function: A function to apply the boundary conditions for
            the simlation. It has the form
            boundary_function(:class:`peridynamics.model.Model`,
            :class:`numpy.ndarray`, `int`). The arguments are the model being
            simulated, the current displacements, and the current step number
            (beginning from 1). `boundary_function` returns a (nnodes, 3)
            :class:`numpy.ndarray` of the updated displacements
            after applying the boundary conditions. Default `None`.
        :type boundary_function: function
        :arg u: The initial displacements for the simulation. If `None` the
            displacements will be initialised to zero. Default `None`.
        :type u: :class:`numpy.ndarray`
        :arg connectivity: The initial connectivity for the simulation. A tuple
            of a neighbour list and the number of neighbours for each node. If
            `None` the connectivity at the time of construction of the
            :class:`Model` object will be used. Default `None`.
        :type connectivity: tuple(:class:`numpy.ndarray`,
            :class:`numpy.ndarray`)
        :arg int first_step: The starting step number. This is useful when
            restarting a simulation, especially if `boundary_function` depends
            on the absolute step number.
        :arg int write: The frequency, in number of steps, to write the system
            to a mesh file by calling :meth:`Model.write_mesh`. If `None` then
            no output is written. Default `None`.
        :arg write_path: The path where the periodic mesh files should be
            written.
        :type write_path: path-like or str

        :returns: A tuple of the final displacements (`u`), damage and
            connectivity.
        :rtype: tuple(:class:`numpy.ndarray`, :class:`numpy.ndarray`,
            tuple(:class:`numpy.ndarray`, :class:`numpy.ndarray`))
        """
        if not isinstance(integrator, Integrator):
            raise InvalidIntegrator(integrator)

        # Create initial displacements is none is provided
        if u is None:
            u = np.zeros((self.nnodes, 3))

        # Use the initial connectivity (when the Model was constructed) if none
        # is provided
        if connectivity is None:
            nlist, n_neigh = self.initial_connectivity
        elif type(connectivity) == tuple:
            if len(connectivity) != 2:
                raise ValueError("connectivity must be of size 2")
            nlist, n_neigh = connectivity
        else:
            raise TypeError("connectivity must be a tuple or None")

        # Create dummy boundary conditions function is none is provided
        if boundary_function is None:
            def boundary_function(model, u, step):
                return u

        # If no write path was provided use the current directory, otherwise
        # ensure write_path is a Path object.
        if write_path is None:
            write_path = pathlib.Path()
        else:
            write_path = pathlib.Path(write_path)

        # Get context and queue
        context = self.context
        queue = self.queue

        # Create constant buffers
        r0_d = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR,
                         hostbuf=self.coords)
        volume_d = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR,
                             hostbuf=self.volume)
        family_d = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR,
                             hostbuf=self.family)

        # Create neighbourlist buffers
        nlist_d = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR,
                            hostbuf=nlist)
        n_neigh_d = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR,
                              hostbuf=n_neigh)

        for step in trange(first_step, first_step+steps,
                           desc="Simulation Progress", unit="steps"):

            # Calculate the force due to bonds on each node
            # f = self._bond_force(u, nlist, n_neigh)
            force = np.empty_like(self.coords)

            # Create buffers
            r_d = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR,
                            hostbuf=self.coords+u)
            force_d = cl.Buffer(context, mf.WRITE_ONLY, force.nbytes)

            # Call kernel
            self.bond_force_kernel(queue, n_neigh.shape, None, r_d, r0_d,
                                   nlist_d,
                                   n_neigh_d, np.int32(self.max_neighbours),
                                   volume_d, np.float64(self.bond_stiffness),
                                   force_d)
            queue.finish()
            cl.enqueue_copy(queue, force, force_d)
            f = force

            # Conduct one integration step
            u = integrator(u, f)
            # Apply boundary conditions
            u = boundary_function(self, u, step)

            # Update neighbour list
            # self._break_bonds(u, nlist, n_neigh)

            # Call kernel
            self.break_bonds_kernel(queue, n_neigh.shape, None, r_d, r0_d,
                                    nlist_d,
                                    n_neigh_d, np.int32(self.max_neighbours),
                                    np.float64(self.critical_strain))
            queue.finish()

            # Calculate the current damage
            # damage = self._damage(n_neigh)
            damage = np.empty(n_neigh.shape, dtype=np.float64)

            # Create buffers
            damage_d = cl.Buffer(context, mf.WRITE_ONLY, damage.nbytes)

            # Call kernel
            self.damage_kernel(queue, damage.shape, None, n_neigh_d, family_d,
                               damage_d)
            queue.finish()

            if write:
                if step % write == 0:
                    cl.enqueue_copy(queue, damage, damage_d)
                    self.write_mesh(write_path/f"U_{step}.vtk", damage, u)

        cl.enqueue_copy(queue, damage, damage_d)
        cl.enqueue_copy(queue, nlist, nlist_d)
        cl.enqueue_copy(queue, n_neigh, n_neigh_d)
        return u, damage, (nlist, n_neigh)


class ContextError(Exception):
    """No suitable context was found by :func:`get_context`."""

    def __init__(self):
        """Exception constructor."""
        message = ("No suitable context was found. You can manually specify"
                   "the context by passing it to ModelCL with the 'context'"
                   "argument.")
        super().__init__(message)

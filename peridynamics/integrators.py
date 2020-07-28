"""Integrators."""
from abc import ABC, abstractmethod
from .cl import double_fp_support, get_context, output_device_info
from pyopencl import mem_flags as mf
from .peridynamics import damage, bond_force, update_displacement, break_bonds
import pyopencl as cl
import pathlib
import numpy as np


class Integrator(ABC):
    """
    Base class for integrators.

    All integrators must define an init method, which may or may not
    use Integrator as a parent class using `super()`. They must also define a
    call method which performs one integration step, a build_special method
    which builds the OpenCL programs which are special to the integrator, and a
    set_special_buffers method which sets the OpenCL buffers which are special
    to the integrator.
    """

    @abstractmethod
    def __init__(self, dt, context=None):
        """
        Create a :class:`Integrator` object.

        This method should be implemennted in every concrete integrator.

        :arg float dt: The length of time (in seconds [s]) of one time-step.
        :arg context: Optional argument for the user to provide a context with
            a single suitable device, default is `None`.
        :type context: :class:`pyopencl._cl.Context` or `NoneType`

        :returns: A :class:`Integrator` object
        """
        self.dt = dt

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
        # Print out device info
        output_device_info(self.context.devices[0])

    @abstractmethod
    def __call__(self):
        """
        Conduct one iteraction of the integrator.

        This method should be implemennted in every concrete integrator.
        """

    @abstractmethod
    def _build_special(self):
        """
        Build OpenCL kernels special to the chosen integrator.

        This method should be implemented in every concrete integrator.
        """

    @abstractmethod
    def _set_special_buffers(self):
        """
        Set buffers that are special to the chosen integrator.

        This method should be implemented in every concrete integrator.
        """

    def build(
            self, nnodes, degrees_freedom, max_neighbours, nregimes, coords,
            volume, family, bc_types, bc_values, force_bc_types,
            force_bc_values):
        """
        Build OpenCL programs.

        Builds the programs that are common to all integrators and the
        buffers which are independent of :class:`Model`.simulation parameters.
        """
        self.nnodes = nnodes
        self.degrees_freedom = degrees_freedom
        self.max_neighbours = max_neighbours
        self.nregimes = nregimes

        kernel_source = open(
            pathlib.Path(__file__).parent.absolute() /
            "cl/peridynamics.cl").read()

        # Build kernels
        self.program = cl.Program(
            self.context, kernel_source).build()
        self.queue = cl.CommandQueue(self.context)

        # Kernels shared between integrators
        self.bond_force_kernel = self.program.bond_force
        self.damage_kernel = self.program.damage

        # Build OpenCL data structures that are independent of
        # :class: Model.simulation parameters
        # Local memory containers for bond forces
        self.local_mem_x = cl.LocalMemory(
            np.dtype(np.float64).itemsize * self.max_neighbours)
        self.local_mem_y = cl.LocalMemory(
            np.dtype(np.float64).itemsize * self.max_neighbours)
        self.local_mem_z = cl.LocalMemory(
            np.dtype(np.float64).itemsize * self.max_neighbours)
        # Local memory container for damage
        self.local_mem = cl.LocalMemory(
            np.dtype(np.float64).itemsize * self.max_neighbours)
        # Read only
        self.r0_d = cl.Buffer(
            self.context, mf.READ_ONLY | mf.COPY_HOST_PTR,
            hostbuf=coords)
        self.vols_d = cl.Buffer(
            self.context, mf.READ_ONLY | mf.COPY_HOST_PTR,
            hostbuf=volume)
        self.family_d = cl.Buffer(
            self.context, mf.READ_ONLY | mf.COPY_HOST_PTR,
            hostbuf=family)
        self.bc_types_d = cl.Buffer(
            self.context, mf.READ_ONLY | mf.COPY_HOST_PTR,
            hostbuf=bc_types)
        self.bc_values_d = cl.Buffer(
            self.context, mf.READ_ONLY | mf.COPY_HOST_PTR,
            hostbuf=bc_values)
        self.force_bc_types_d = cl.Buffer(
            self.context, mf.READ_ONLY | mf.COPY_HOST_PTR,
            hostbuf=force_bc_types)
        self.force_bc_values_d = cl.Buffer(
            self.context, mf.READ_ONLY | mf.COPY_HOST_PTR,
            hostbuf=force_bc_values)

        # Build programs that are special to the chosen integrator
        self._build_special()

    def set_buffers(
            self, nlist, bond_stiffness, critical_stretch, plus_cs, u, ud,
            force, damage, regimes):
        """
        Initialise the OpenCL buffers.

        Initialises just the buffers which are dependent on
        :class:`Model`.simulation parameters.
        """
        self.bond_stiffness = bond_stiffness
        self.critical_stretch = critical_stretch
        self.regimes = regimes

        # Build OpenCL data structures that are dependent on
        # :class: Model.simulation parameters
        # Read and write
        self.force_d = cl.Buffer(
            self.context, mf.READ_WRITE, force.nbytes)
        self.nlist_d = cl.Buffer(
            self.context, mf.READ_WRITE | mf.COPY_HOST_PTR,
            hostbuf=nlist)
        self.u_d = cl.Buffer(
            self.context, mf.READ_WRITE | mf.COPY_HOST_PTR,
            hostbuf=u)
        self.ud_d = cl.Buffer(
            self.context, mf.READ_WRITE | mf.COPY_HOST_PTR,
            hostbuf=ud)
        # Write only
        self.damage_d = cl.Buffer(
            self.context, mf.WRITE_ONLY, damage.nbytes)

        self._set_special_buffers()

    def _damage(self, n_list_d, family_d, n_neigh_d, damage_d, local_mem):
        """Calculate bond damage."""
        queue = self.queue

        # Call kernel
        self.damage_kernel(
            queue, (self.nnodes * self.max_neighbours,),
            (self.max_neighbours,), n_list_d, family_d, n_neigh_d, damage_d,
            local_mem)
        queue.finish()

    def _bond_force(
            self, u_d, force_d, r0_d, vols_d, nlist_d,
            force_bc_types_d, force_bc_values_d, local_mem_x, local_mem_y,
            local_mem_z, force_load_scale, bond_stiffness, critical_stretch):
        """Calculate the force due to bonds acting on each node."""
        queue = self.queue
        # Call kernel
        self.bond_force_kernel(
                queue, (self.nnodes * self.max_neighbours,),
                (self.max_neighbours,), u_d, force_d, r0_d, vols_d, nlist_d,
                force_bc_types_d, force_bc_values_d, local_mem_x,
                local_mem_y, local_mem_z, np.float64(force_load_scale),
                np.float64(bond_stiffness), np.float64(critical_stretch))
        queue.finish()

    def write(self, u, ud, force, damage, nlist, n_neigh):
        """Copy the state variables from device memory to host memory."""
        queue = self.queue
        # Calculate the damage
        self._damage(self.nlist_d, self.family_d, self.n_neigh_d,
                     self.damage_d, self.local_mem)

        cl.enqueue_copy(queue, damage, self.damage_d)
        cl.enqueue_copy(queue, u, self.u_d)
        cl.enqueue_copy(queue, ud, self.ud_d)
        cl.enqueue_copy(queue, force, self.force_d)
        cl.enqueue_copy(queue, nlist, self.nlist_d)
        cl.enqueue_copy(queue, n_neigh, self.n_neigh_d)
        return (u, ud, force, damage, nlist, n_neigh)


class Euler(Integrator):
    r"""
    Euler integrator for cython.

    The Euler method is a first-order numerical integration method. The
    integration is given by,

    .. math::
        u(t + \delta t) = u(t) + \delta t f(t)

    where :math:`u(t)` is the displacement at time :math:`t`, :math:`f(t)` is
    the force at time :math:`t`, :math:`\delta t` is the time step and
    """

    def __init__(self, dt):
        """
        Create an :class:`Euler` integrator object.

        :returns: A :class:`Euler` object
        """
        self.dt = dt
        # Not an OpenCL integrator
        self.context = None

    def __call__(self, displacement_bc_scale, force_bc_scale):
        """Conduct one iteration of the integrator."""
        # Calculate the force due to bonds on each node
        self.force = self._bond_force(force_bc_scale)

        # Conduct one integration step
        self._update_displacement(self.force, displacement_bc_scale)

        # Update neighbour list
        self._break_bonds()

    def set_buffers(
            self, nlist, n_neigh, bond_stiffness, critical_stretch, plus_cs,
            u, force, damage, regimes):
        """
        Initiate arrays that are dependent on simulation parameters.

        Since :class:`Euler` uses cython in place of OpenCL, there are no
        buffers to be set, just :class:`numpy.ndarray` arrays.
        """
        self.nlist = nlist
        self.n_neigh = n_neigh
        self.bond_stiffness = bond_stiffness
        self.critical_stretch = critical_stretch
        self.u = u
        self.force = force

    def build(
            self, nnodes, degrees_freedom, max_neighbours, nregimes, coords,
            volume, family, bc_types, bc_values, force_bc_types,
            force_bc_values):
        """
        Initiate integrator arrays.

        Since :class:`Euler` uses cython in place of OpenCL, there are no
        programs to be built. Builds the arrays that are common to all
        integrators and the buffers which are independent of
        :class:`Model`.simulation parameters.
        """
        self.nnodes = nnodes
        self.coords = coords
        self.family = family
        self.volume = volume
        self.bc_types = bc_types
        self.bc_values = bc_values
        self.force_bc_types = force_bc_types
        self.force_bc_values = force_bc_values

    def _set_special_buffers(self):
        """Set buffers programs that are special to the Euler integrator."""

    def _build_special(self):
        """Build programs that are special to the Euler integrator."""

    def _update_displacement(self, force, displacement_bc_scale):
        update_displacement(
            self.u, self.bc_values, self.bc_types, force,
            displacement_bc_scale, self.dt)

    def _break_bonds(self):
        """Break bonds which have exceeded the critical strain."""
        break_bonds(self.coords+self.u, self.coords, self.nlist, self.n_neigh,
                    self.critical_stretch)

    def _damage(self):
        """Calculate bond damage."""
        return damage(self.n_neigh, self.family)

    def _bond_force(self, force_bc_scale):
        """Calculate the force due to bonds acting on each node."""
        force = bond_force(
            self.coords+self.u, self.coords, self.nlist, self.n_neigh,
            self.volume, self.bond_stiffness, self.force_bc_values,
            self.force_bc_types, force_bc_scale)
        return force

    def write(self, damage, u, force, nlist, n_neigh):
        """Return the state variable arrays."""
        damage = self._damage()
        return (damage, self.u, self.force, self.nlist, self.n_neigh)


class EulerOpenCL(Integrator):
    r"""
    Euler integrator for OpenCL.

    The Euler method is a first-order numerical integration method. The
    integration is given by,

    .. math::
        u(t + \delta t) = u(t) + \delta t f(t) d

    where :math:`u(t)` is the displacement at time :math:`t`, :math:`f(t)` is
    the force at time :math:`t`, :math:`\delta t` is the time step and
    """

    def __init__(self, *args, **kwargs):
        """
        Create an :class:`Euler` integrator object.

        :returns: A :class:`Euler` object
        """
        super().__init__(*args, **kwargs)

    def __call__(self, displacement_bc_scale, force_bc_scale):
        """Conduct one iteration of the integrator."""
        self._update_displacement(
            self.force_d, self.u_d, self.bc_types_d, self.bc_values_d,
            displacement_bc_scale, self.dt)
        self._bond_force(
            self.u_d, self.force_d, self.r0_d, self.vols_d, self.nlist_d,
            self.n_neigh_d, self.force_bc_types_d, self.force_bc_values_d,
            self.local_mem_x, self.local_mem_y, self.local_mem_z,
            force_bc_scale, self.bond_stiffness, self.critical_stretch)

    def _build_special(self):
        """Build OpenCL kernels special to the Euler integrator."""
        kernel_source = open(
            pathlib.Path(__file__).parent.absolute() /
            "cl/euler.cl").read()

        # Build kernels
        self.euler = cl.Program(
            self.context, kernel_source).build()
        self.update_displacement_kernel = self.euler.update_displacement

    def _set_special_buffers(self):
        """Set buffers special to the Euler integrator."""

    def _update_displacement(
            self, force_d, u_d, bc_types_d, bc_values_d,
            displacement_load_scale, dt):
        """Update displacements."""
        queue = self.queue
        # Call kernel
        self.update_displacement_kernel(
                self.queue, (self.degrees_freedom * self.nnodes,), None,
                force_d, u_d, bc_types_d, bc_values_d,
                np.float64(displacement_load_scale), np.float64(dt))
        queue.finish()
        return u_d


class ContextError(Exception):
    """No suitable context was found by :func:`get_context`."""

    def __init__(self):
        """Exception constructor."""
        message = ("No suitable context was found. You can manually specify"
                   "the context by passing it to ModelCL with the 'context'"
                   "argument.")
        super().__init__(message)

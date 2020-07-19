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

    All integrators must define a call method which performs one
    integration step.
    """
    def __init__(self, dt, density, damping=1.0):
        """
        Create a :class:`Integrator` object.

        :arg float dt: The length of time (in seconds [s]) of one time-step.
        :arg float damping: The damping factor. The default is 1.0
        :arg float density: The density of the bulk material in kg/m^3.

        :returns: A :class:`Integrator` object
        """
        self.dt = dt
        self.damping = damping
        self.density = density

    def build_program(
            self, nnodes, degrees_freedom, max_neighbours, nregimes, coords,
            volume, family, bc_types, bc_values, force_bc_types,
            force_bc_values, context):
        """
        Builds the programs that are common to all integrators and the
        buffers which are independent of :class:`Model`.simulation parameters.
        """
        self.nnodes = nnodes
        self.degrees_freedom = degrees_freedom
        self.max_neighbours = max_neighbours
        self.nregimes = nregimes

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

        kernel_source = open(
            pathlib.Path(__file__).parent.absolute() / \
                "cl/peridynamics.cl").read()
        # JIT Compiler command line arguments
        SEP = " "
        options_string = (
            "-cl-fast-relaxed-math" + SEP
            + "-Ddof=" + str(self.degrees_freedom) + SEP
            + "-Ddof_nnodes=" + str(self.degrees_freedom * self.nnodes) + SEP
            + "-Dnnodes=" + str(self.nnodes) + SEP
            + "-Dmax_neighbours=" + str(self.max_neighbours) + SEP
            + "-Ddt=" + str(self.dt) + SEP
            + "-Dnregimes=" + str(self.nregimes) + SEP)

        # Build kernels
        self.program = cl.Program(
            self.context, kernel_source).build([options_string])
        self.queue = cl.CommandQueue(self.context)

        # Kernels shared between integrators
        self.bond_force_kernel = self.program.bond_force
        self.damage_kernel = self.program.damage_new

        # Build OpenCL data structures that are independent of
        # :class: Model.simulation parameters
        # Local memory containers for Bond forces
        self.local_mem_x = cl.LocalMemory(
            np.dtype(np.float64).itemsize * self.max_neighbours)
        self.local_mem_y = cl.LocalMemory(
            np.dtype(np.float64).itemsize * self.max_neighbours)
        self.local_mem_z = cl.LocalMemory(
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

        self.build_specific_program()

    def initialise_buffers(
            self, nlist, n_neigh, bond_stiffness,
            critical_stretch, plus_cs, u, ud, damage):
        """
        Initialise the OpenCL buffers that are dependent on simulation
        parameters.

        """
        self.bond_stiffness = bond_stiffness
        self.critical_stretch = critical_stretch

        # Build OpenCL data structures that are dependent on
        # :class: Model.simulation parameters
        
        # Read and write
        self.nlist_d = cl.Buffer(
            self.context, mf.READ_WRITE | mf.COPY_HOST_PTR,
            hostbuf=nlist)
        self.n_neigh_d = cl.Buffer(
            self.context, mf.READ_WRITE | mf.COPY_HOST_PTR,
            hostbuf=n_neigh)
        self.u_d = cl.Buffer(
            self.context, mf.READ_WRITE, u.nbytes)
        self.ud_d = cl.Buffer(
            self.context, mf.READ_WRITE, ud.nbytes)

        # Write only
        self.damage_d = cl.Buffer(
            self.context, mf.WRITE_ONLY, damage.nbytes)
        
        self.initialise_specific_buffers()

    @abstractmethod
    def __call__(self):
        """
        Conduct one iteraction of the integrator.

        This method should be implemennted in every concrete integrator.
        """

    @abstractmethod
    def initialise_specific_buffers(self):
        """
        Initialise buffers specific to the chosen integrator.

        This method should be implemennted in every concrete integrator.
        """

    @abstractmethod
    def build_specific_program(self):
        """
        Initialise buffers specific to the chosen integrator.

        This method should be implemennted in every concrete integrator.
        """

    def _damage(self, n_neigh_d, family_d, damage_d):
        """Calculate bond damage."""
        queue = self.queue

        # Call kernel
        self.damage_kernel(queue, (self.nnodes,), None, n_neigh_d,
                               family_d, damage_d)
        queue.finish()

    def write(self, damage, u, ud, nlist, n_neigh):
        """Copy the state variables from device memory to host memory."""
        queue = self.queue
        # Calculate the damage
        self._damage(self.n_neigh_d, self.family_d, self.damage_d)

        cl.enqueue_copy(queue, damage, self.damage_d)
        cl.enqueue_copy(queue, u, self.u_d)
        cl.enqueue_copy(queue, ud, self.ud_d)
        cl.enqueue_copy(queue, nlist, self.nlist_d)
        cl.enqueue_copy(queue, n_neigh, self.n_neigh_d)
        return (damage, u, ud, nlist, n_neigh)

class Euler(Integrator):
    r"""
    Euler integrator.

    The Euler method is a first-order numerical integration method. The
    integration is given by,

    .. math::
        u(t + \delta t) = u(t) + \delta t f(t) d

    where :math:`u(t)` is the displacement at time :math:`t`, :math:`f(t)` is
    the force at time :math:`t`, :math:`\delta t` is the time step and
    :math:`d` is a damping factor.
    """

    def __init__(self, *args, **kwargs):
        """
        Create an :class:`Euler` integrator object.

        :returns: A :class:`Euler` object
        """
        super().__init__(*args, **kwargs)

    def __call__(self, displacement_bc_scale, force_bc_scale):
        """
        Conduct one iteration of the integrator.

        :arg u: A (`nnodes`, 3) array containing the displacements of all
            nodes.
        :type u: :class:`numpy.ndarray`
        :arg f: A (`nnodes`, 3) array containing the components of the force
            acting on each node.
        :type f: :class:`numpy.ndarray`

        :returns: The new displacements after integration.
        :rtype: :class:`numpy.ndarray`
        """
        # Calculate the force due to bonds on each node
        self.ud = self._bond_force(force_bc_scale)

        # Conduct one integration step
        self._update_displacement(self.ud, displacement_bc_scale)

        # Update neighbour list
        self._break_bonds()

    def initialise_buffers(self, nlist, n_neigh, bond_stiffness,
            critical_stretch, plus_cs, u, ud, damage):

        self.nlist = nlist
        self.n_neigh = n_neigh
        self.bond_stiffness = bond_stiffness
        self.critical_stretch = critical_stretch
        self.u = u
        self.ud = ud

    def build_program(
            self, nnodes, degrees_freedom, max_neighbours, nregimes, coords,
            volume, family, bc_types, bc_values, force_bc_types,
            force_bc_values, context):

        self.nnodes = nnodes
        self.coords = coords
        self.family = family
        self.volume = volume
        self.bc_types = bc_types
        self.bc_values = bc_values
        self.force_bc_types = force_bc_types
        self.force_bc_values = force_bc_values

    def initialise_specific_buffers(self):
        """
        Initialise buffers and programs that are specific to the integrator.
        """

    def build_specific_program(self):
        """
        Initialise buffers specific to the chosen integrator.

        This method should be implemennted in every concrete integrator.
        """

    def _update_displacement(self, ud, displacement_bc_scale):
        update_displacement(
            self.u, self.bc_values, self.bc_types, ud, 
            displacement_bc_scale, self.dt)

    def _break_bonds(self):
        """
        Break bonds which have exceeded the critical strain.
        """
        break_bonds(self.coords+self.u, self.coords, self.nlist, self.n_neigh,
                    self.critical_stretch)

    def _damage(self):
        """
        Calculate bond damage.
        """
        return damage(self.n_neigh, self.family)

    def _bond_force(self, force_bc_scale):
        """
        Calculate the force due to bonds acting on each node.
        """
        ud = bond_force(self.coords+self.u, self.coords, self.nlist,
                             self.n_neigh, self.volume, self.bond_stiffness,
                             self.force_bc_values, self.force_bc_types,
                             force_bc_scale)
        return ud

    def write(self, damage, u, ud, nlist, n_neigh):
        # Calculate the current damage
        damage = self._damage()
        return damage, self.u, self.ud, self.nlist, self.n_neigh

class EulerOpenCL(Integrator):
    r"""
    Euler integrator for OpenCL.

    The Euler method is a first-order numerical integration method. The
    integration is given by,

    .. math::
        u(t + \delta t) = u(t) + \delta t f(t) d

    where :math:`u(t)` is the displacement at time :math:`t`, :math:`f(t)` is
    the force at time :math:`t`, :math:`\delta t` is the time step and
    :math:`d` is a damping factor.
    """

    def __init__(self, *args, **kwargs):
        """
        Create an :class:`Euler` integrator object.

        :arg float dt: The integration time step.
        :arg float damping: The damping factor. The default is 1.0

        :returns: A :class:`Euler` object
        """
        super().__init__(*args, **kwargs)

    def __call__(self, displacement_bc_scale, force_bc_scale):
        """
        Conduct one iteration of the integrator.

        :returns: None
        :rtype: NoneType
        """
        self._update_displacement(
            self.ud_d, self.u_d, self.bc_types_d, self.bc_values_d, 
            displacement_bc_scale, self.dt)
        self._bond_force(
            self.u_d, self.ud_d, self.r0_d, self.vols_d, self.nlist_d,
            self.n_neigh_d, self.force_bc_types_d, self.force_bc_values_d,
            self.local_mem_x, self.local_mem_y, self.local_mem_z,
            force_bc_scale, self.bond_stiffness, self.critical_stretch)

    def build_specific_program(self):
        """
        Initialise buffers and programs that are specific to the integrator.
        """
        kernel_source = open(
            pathlib.Path(__file__).parent.absolute() / \
                "cl/euler.cl").read()

        # JIT Compiler command line arguments
        SEP = " "
        options_string = (
            "-cl-fast-relaxed-math" + SEP
            + "-Ddof_nnodes=" + str(self.degrees_freedom * self.nnodes) + SEP
            + "-Dnnodes=" + str(self.nnodes) + SEP
            + "-Ddof=" + str(self.degrees_freedom) + SEP)

        # Build kernels
        self.euler = cl.Program(
            self.context, kernel_source).build([options_string])
        self.update_displacement_kernel = self.euler.update_displacement
        self.bond_force_kernel = self.euler.bond_force

    def initialise_specific_buffers(self):
        """
        Initialise buffers and programs that are specific to the integrator.
        """

    def _update_displacement(
            self, ud_d, u_d, bc_types_d, bc_values_d, displacement_load_scale,
            dt):
        """Update displacements."""
        queue = self.queue
        # Call kernel
        self.update_displacement_kernel(
                self.queue, (self.degrees_freedom * self.nnodes,), None,
                ud_d, u_d, bc_types_d, bc_values_d, 
                np.float64(displacement_load_scale), np.float64(dt))
        queue.finish()
        return u_d

    def _bond_force(
            self, u_d, ud_d, r0_d, vols_d, nlist_d, n_neigh_d,
            force_bc_types_d, force_bc_values_d, local_mem_x, local_mem_y,
            local_mem_z, force_load_scale, bond_stiffness, critical_stretch):
        """Calculate the force due to bonds acting on each node."""
        queue = self.queue
        # Call kernel
        self.bond_force_kernel(
                queue, (self.nnodes * self.max_neighbours,),
                (self.max_neighbours,), u_d, ud_d, r0_d, vols_d, nlist_d,
                n_neigh_d, force_bc_types_d, force_bc_values_d, local_mem_x,
                local_mem_y, local_mem_z, np.float64(force_load_scale),
                np.float64(bond_stiffness), np.float64(critical_stretch))
        queue.finish()


class ContextError(Exception):
    """No suitable context was found by :func:`get_context`."""

    def __init__(self):
        """Exception constructor."""
        message = ("No suitable context was found. You can manually specify"
                   "the context by passing it to ModelCL with the 'context'"
                   "argument.")
        super().__init__(message)

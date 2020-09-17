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

    All integrators must define an init method, which may or may not inherit
    Integrator as a parent class using `super()`. All integrators that inherit
    Integrator are OpenCL implementations that can use GPU or CPU. All
    integrators must also define a call method which performs one integration
    step, a `_build_special` method which builds the OpenCL programs which are
    special to the integrator, and a `_create_special_buffers` method which
    creates the OpenCL buffers which are special to the integrator.
    """

    @abstractmethod
    def __init__(self, dt, context=None):
        """
        Create an :class:`Integrator` object.

        This method should be implemented in every concrete integrator.

        :arg float dt: The length of time (in seconds [s]) of one time-step.
        :arg context: Optional argument for the user to provide a context with
            a single suitable device, default is None.
        :type context: :class:`pyopencl._cl.Context` or NoneType

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

        self.queue = cl.CommandQueue(self.context)

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
    def _create_special_buffers(self):
        """
        Create buffers that are special to the chosen integrator.

        This method should be implemented in every concrete integrator.
        """

    def build(
            self, nnodes, degrees_freedom, max_neighbours, coords, volume,
            family, bc_types, bc_values, force_bc_types, force_bc_values,
            stiffness_corrections, bond_types, densities):
        """
        Build OpenCL programs.

        Builds the programs that are common to all integrators and the
        buffers which are independent of
        :meth:`peripy.model.Model.simulate` parameters.
        """
        self.nnodes = nnodes
        self.degrees_freedom = degrees_freedom
        self.max_neighbours = max_neighbours
        self.densities = densities

        kernel_source = open(
            pathlib.Path(__file__).parent.absolute() /
            "cl/peridynamics.cl").read()

        # Build kernels
        self.program = cl.Program(
            self.context, kernel_source).build()

        # Build bond_force program
        if (stiffness_corrections is None) and (bond_types is None):
            self.bond_force_kernel = self.program.bond_force1
            # Placeholder buffers
            stiffness_corrections = np.array([0], dtype=np.float64)
            bond_types = np.array([0], dtype=np.intc)
            self.stiffness_corrections_d = cl.Buffer(
                self.context, mf.READ_ONLY | mf.COPY_HOST_PTR,
                hostbuf=stiffness_corrections)
            self.bond_types_d = cl.Buffer(
                self.context, mf.READ_ONLY | mf.COPY_HOST_PTR,
                hostbuf=bond_types)
        elif (stiffness_corrections is not None) and (bond_types is None):
            self.bond_force_kernel = self.program.bond_force2
            self.stiffness_corrections_d = cl.Buffer(
                self.context, mf.READ_ONLY | mf.COPY_HOST_PTR,
                hostbuf=stiffness_corrections)
            # Placeholder buffers
            bond_types = np.array([0], dtype=np.intc)
            self.bond_types_d = cl.Buffer(
                self.context, mf.READ_ONLY | mf.COPY_HOST_PTR,
                hostbuf=bond_types)
        elif (stiffness_corrections is None) and (bond_types is not None):
            self.bond_force_kernel = self.program.bond_force3
            self.bond_types_d = cl.Buffer(
                self.context, mf.READ_ONLY | mf.COPY_HOST_PTR,
                hostbuf=bond_types)
            # Placeholder buffers
            stiffness_corrections = np.array([0], dtype=np.float64)
            self.stiffness_corrections_d = cl.Buffer(
                self.context, mf.READ_ONLY | mf.COPY_HOST_PTR,
                hostbuf=stiffness_corrections)
        elif ((stiffness_corrections is not None)
              and (bond_types is not None)):
            self.bond_force_kernel = self.program.bond_force4
            self.stiffness_corrections_d = cl.Buffer(
                self.context, mf.READ_ONLY | mf.COPY_HOST_PTR,
                hostbuf=stiffness_corrections)
            self.bond_types_d = cl.Buffer(
                self.context, mf.READ_ONLY | mf.COPY_HOST_PTR,
                hostbuf=bond_types)

        self.damage_kernel = self.program.damage

        # Create OpenCL buffers that are independent of
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

    def create_buffers(
            self, nlist, n_neigh, bond_stiffness, critical_stretch, plus_cs, u,
            ud, udd, force, body_force, damage, regimes, nregimes,
            nbond_types):
        """
        Initialise the OpenCL buffers.

        Initialises only the buffers which are dependent on
        :meth:`peripy.model.Model.simulate` parameters.
        """
        if (nbond_types == 1) and (nregimes == 1):
            self.bond_stiffness_d = np.float64(bond_stiffness)
            self.critical_stretch_d = np.float64(critical_stretch)
            # Placeholder buffers
            plus_cs = np.array([0], dtype=np.float64)
            regimes = np.array([0], dtype=np.intc)
            self.plus_cs_d = cl.Buffer(
                self.context, mf.READ_WRITE | mf.COPY_HOST_PTR,
                hostbuf=plus_cs)
            self.regimes_d = cl.Buffer(
                self.context, mf.READ_WRITE | mf.COPY_HOST_PTR,
                hostbuf=regimes)
        else:
            self.bond_stiffness_d = cl.Buffer(
                self.context, mf.READ_ONLY | mf.COPY_HOST_PTR,
                hostbuf=bond_stiffness)
            self.critical_stretch_d = cl.Buffer(
                self.context, mf.READ_ONLY | mf.COPY_HOST_PTR,
                hostbuf=critical_stretch)
            self.plus_cs_d = cl.Buffer(
                self.context, mf.READ_WRITE | mf.COPY_HOST_PTR,
                hostbuf=plus_cs)
            self.regimes_d = cl.Buffer(
                self.context, mf.READ_WRITE | mf.COPY_HOST_PTR,
                hostbuf=regimes)

        self.nregimes = np.intc(nregimes)
        self.nbond_types = np.intc(nbond_types)

        # Create OpenCL buffers that are dependent on
        # :meth:`peripy.model.Model.simulate` parameters.
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
        self.udd_d = cl.Buffer(
            self.context, mf.READ_WRITE | mf.COPY_HOST_PTR,
            hostbuf=udd)
        # Write only
        self.damage_d = cl.Buffer(
            self.context, mf.WRITE_ONLY, damage.nbytes)
        self.body_force_d = cl.Buffer(
            self.context, mf.WRITE_ONLY, body_force.nbytes)
        self.n_neigh_d = cl.Buffer(
            self.context, mf.WRITE_ONLY, n_neigh.nbytes)

        self._create_special_buffers()

    def _damage(self, nlist_d, family_d, n_neigh_d, damage_d, local_mem):
        """Calculate bond damage."""
        queue = self.queue
        # Call kernel
        self.damage_kernel(
            queue, (self.nnodes * self.max_neighbours,),
            (self.max_neighbours,), nlist_d, family_d, n_neigh_d, damage_d,
            local_mem)
        queue.finish()

    def _bond_force(
            self, u_d, force_d, body_force_d, r0_d, vols_d, nlist_d,
            force_bc_types_d, force_bc_values_d, stiffness_corrections_d,
            bond_types_d, regimes_d, plus_cs_d, local_mem_x, local_mem_y,
            local_mem_z, bond_stiffness_d, critical_stretch_d,
            force_bc_magnitude, nregimes):
        """Calculate the force due to bonds acting on each node."""
        queue = self.queue
        # Call kernel
        self.bond_force_kernel(
                queue, (self.nnodes * self.max_neighbours,),
                (self.max_neighbours,), u_d, force_d, body_force_d, r0_d,
                vols_d, nlist_d, force_bc_types_d, force_bc_values_d,
                stiffness_corrections_d, bond_types_d, regimes_d, plus_cs_d,
                local_mem_x, local_mem_y, local_mem_z, bond_stiffness_d,
                critical_stretch_d, np.float64(force_bc_magnitude),
                np.intc(nregimes))
        queue.finish()

    def write(self, u, ud, udd, force, body_force, damage, nlist, n_neigh):
        """Copy the state variables from device memory to host memory."""
        queue = self.queue
        # Calculate the damage
        self._damage(self.nlist_d, self.family_d, self.n_neigh_d,
                     self.damage_d, self.local_mem)

        cl.enqueue_copy(queue, damage, self.damage_d)
        cl.enqueue_copy(queue, u, self.u_d)
        cl.enqueue_copy(queue, ud, self.ud_d)
        cl.enqueue_copy(queue, udd, self.udd_d)
        cl.enqueue_copy(queue, force, self.force_d)
        cl.enqueue_copy(queue, body_force, self.body_force_d)
        cl.enqueue_copy(queue, nlist, self.nlist_d)
        cl.enqueue_copy(queue, n_neigh, self.n_neigh_d)
        return (u, ud, udd, force, body_force, damage, nlist, n_neigh)


class Euler(Integrator):
    r"""
    Euler integrator for cython.

    C implementation of the Euler integrator generated using Cython. Uses CPU
    only. The Euler method is a first-order numerical integration method. The
    integration is given by,

    .. math::
        u(t + \delta t) = u(t) + \delta t f(t),

    where :math:`u(t)` is the displacement at time :math:`t`, :math:`f(t)` is
    the force density at time :math:`t`, :math:`\delta t` is the time step.
    """

    def __init__(self, dt):
        """
        Create an :class:`Euler` integrator object.

        :arg float dt: The length of time (in seconds [s]) of one time-step.

        :returns: An :class:`Euler` object
        """
        self.dt = dt
        # Not an OpenCL integrator
        self.context = None

    def __call__(self, displacement_bc_magnitude, force_bc_magnitude):
        """
        Conduct one iteration of the integrator.

        :arg float displacement_bc_magnitude: the magnitude applied to the
             displacement boundary conditions for the current time-step.
        :arg float force_bc_magnitude: the magnitude applied to the force
            boundary conditions for the current time-step.
        """
        # Update neighbour list
        self._break_bonds(
            self.u, self.nlist, self.n_neigh)

        # Calculate the force due to bonds on each node
        self.force = self._bond_force(
            force_bc_magnitude, self.u, self.nlist, self.n_neigh)

        # Conduct one integration step
        self._update_displacement(
            self.u, self.force, displacement_bc_magnitude)

    def create_buffers(
            self, nlist, n_neigh, bond_stiffness, critical_stretch, plus_cs,
            u, ud, udd, force, body_force, damage, regimes, nregimes,
            nbond_types):
        """
        Initiate arrays that are dependent on simulation parameters.

        Initiates arrays that are dependent on
        :meth:`peripy.model.Model.simulate` parameters. Since
        :class:`Euler` uses cython in place of OpenCL, there are no
        buffers to be created, just python objects that are used as arguments
        of the cython functions.
        """
        if nregimes != 1:
            raise ValueError("n-linear damage model's are not supported by "
                             "this integrator. Please supply just one "
                             "bond_stiffness.")
        if nbond_types != 1:
            raise ValueError("n-material composite models are not supported by"
                             " this integrator. Please supply just one "
                             "material type and bond_stiffness.")
        self.nlist = nlist
        self.n_neigh = n_neigh
        self.bond_stiffness = bond_stiffness
        self.critical_stretch = critical_stretch
        self.u = u
        self.ud = ud
        self.udd = udd
        self.force = force
        self.body_force = body_force

    def build(
            self, nnodes, degrees_freedom, max_neighbours, coords, volume,
            family, bc_types, bc_values, force_bc_types, force_bc_values,
            stiffness_corrections, bond_types, densities):
        """
        Initiate integrator arrays.

        Since :class:`Euler` uses cython in place of OpenCL, there are no
        OpenCL programs or buffers to be built/created. Instead, this method
        instantiates the arrays and variables that are independent of
        :meth:`peripy.model.Model.simulate` parameters as python
        objects that are used as arguments of the cython functions.
        """
        self.nnodes = nnodes
        self.coords = coords
        self.family = family
        self.volume = volume
        self.bc_types = bc_types
        self.bc_values = bc_values
        self.force_bc_types = force_bc_types
        self.force_bc_values = force_bc_values
        if bond_types is not None:
            raise ValueError("bond_types are not supported by this "
                             "integrator (expected {}, got {}), please use "
                             "EulerCL instead".format(
                                 type(None),
                                 type(bond_types)))
        if stiffness_corrections is not None:
            raise ValueError("stiffness_corrections are not supported by this "
                             "integrator (expected {}, got {}), please use "
                             "EulerCL instead".format(
                                 type(None),
                                 type(stiffness_corrections)))
        if densities is not None:
            raise ValueError("densities are not supported by this "
                             "integrator (expected {}, got {}). This "
                             " integrator neglects inertial effects. Do not "
                             "supply a density or is_density argument or, "
                             "alternatively, please use a dynamic integrator, "
                             "such as EulerCromerCL.".format(
                                 type(None),
                                 type(densities)))

    def _create_special_buffers(self):
        """Create buffers programs that are special to the Euler integrator."""
        # There are none

    def _build_special(self):
        """Build programs that are special to the Euler integrator."""
        # There are none

    def _update_displacement(self, u, force, displacement_bc_magnitude):
        update_displacement(
            u, self.bc_values, self.bc_types, force, displacement_bc_magnitude,
            self.dt)

    def _break_bonds(self, u, nlist, n_neigh):
        """Break bonds which have exceeded the critical strain."""
        break_bonds(self.coords+u, self.coords, nlist, n_neigh,
                    self.critical_stretch)

    def _damage(self, n_neigh):
        """Calculate bond damage."""
        return damage(n_neigh, self.family)

    def _bond_force(self, force_bc_magnitude, u, nlist, n_neigh):
        """Calculate the force due to bonds acting on each node."""
        force = bond_force(
            self.coords+u, self.coords, nlist, n_neigh,
            self.volume, self.bond_stiffness, self.force_bc_values,
            self.force_bc_types, force_bc_magnitude)
        return force

    def write(self, damage, u, ud, udd, force, body_force, nlist, n_neigh):
        """Return the state variable arrays."""
        damage = self._damage(self.n_neigh)
        return (self.u, self.ud, self.udd, self.force, self.body_force, damage,
                self.nlist, self.n_neigh)


class EulerCL(Integrator):
    r"""
    Euler integrator for OpenCL.

    The Euler method is a first-order numerical integration method. The
    integration is given by,

    .. math::
        u(t + \delta t) = u(t) + \delta t f(t),

    where :math:`u(t)` is the displacement at time :math:`t`, :math:`f(t)` is
    the force density at time :math:`t`, :math:`\delta t` is the time step.
    """

    def __init__(self, *args, **kwargs):
        """
        Create an :class:`EulerCL` integrator object.

        :returns: An :class:`EulerCL` object
        """
        super().__init__(*args, **kwargs)

    def __call__(self, displacement_bc_magnitude, force_bc_magnitude):
        """
        Conduct one iteration of the integrator.

        :arg float displacement_bc_magnitude: the magnitude applied to the
             displacement boundary conditions for the current time-step.
        :arg float force_bc_magnitude: the magnitude applied to the force
            boundary conditions for the current time-step.
        """
        self._bond_force(
            self.u_d, self.force_d, self.body_force_d, self.r0_d, self.vols_d,
            self.nlist_d, self.force_bc_types_d, self.force_bc_values_d,
            self.stiffness_corrections_d, self.bond_types_d, self.regimes_d,
            self.plus_cs_d, self.local_mem_x, self.local_mem_y,
            self.local_mem_z, self.bond_stiffness_d, self.critical_stretch_d,
            force_bc_magnitude, self.nregimes)

        self._update_displacement(
            self.force_d, self.u_d, self.bc_types_d, self.bc_values_d,
            displacement_bc_magnitude, self.dt)

    def _build_special(self):
        """Build OpenCL kernels special to the Euler integrator."""
        kernel_source = open(
            pathlib.Path(__file__).parent.absolute() /
            "cl/euler.cl").read()

        if self.densities is not None:
            raise ValueError("densities are not supported by this "
                             "integrator (expected {}, got {}). This "
                             " integrator neglects inertial effects. Do not "
                             "supply a density or is_density argument or, "
                             "alternatively, use a dynamic integrator, "
                             "such as EulerCromerCL.".format(
                                 type(None),
                                 type(self.densities)))

        # Build kernels
        self.euler = cl.Program(
            self.context, kernel_source).build()
        self.update_displacement_kernel = self.euler.update_displacement

    def _create_special_buffers(self):
        """Create buffers special to the Euler integrator."""
        # There are none

    def _update_displacement(
            self, force_d, u_d, bc_types_d, bc_values_d,
            displacement_bc_magnitude, dt):
        """Update displacements."""
        queue = self.queue
        # Call kernel
        self.update_displacement_kernel(
                self.queue, (self.degrees_freedom * self.nnodes,), None,
                force_d, u_d, bc_types_d, bc_values_d,
                np.float64(displacement_bc_magnitude), np.float64(dt))
        queue.finish()
        return u_d


class EulerCromerCL(Integrator):
    r"""
    Euler Cromer integrator for OpenCL which can use GPU or CPU.

    The Euler-Cromer method is a first-order numerical integration method. The
    integration is given by,

    .. math::
        \dot{u}(t + \delta t) = \dot{u}(t) + \delta t \ddot{u}(t),
    .. math::
        u(t + \delta t) = u(t) + \delta t \dot{u}(t + \delta t),

    where :math:`u(t)` is the displacement at time :math:`t`,
    :math:`\dot{u}(t)` is the velocity at time :math:`t`, :math:`\ddot{u}(t)`
    is the acceleration at time :math:`t`, and :math:`\delta t` is the time
    step.

    A dynamic relaxation damping term :math:`\eta \dot{u}(t)` is added to the
    equation of motion so that the solution to quickly converges to a steady
    state solution in quasi-static problems. Given the velocity and
    displacement vectors of each node at time step :math:`t`, the acceleration
    at time step :math:`t` is given by the equation of motion,

    .. math::
        \ddot{u}(t) = \frac{f(t) - \eta \dot{u}(t)}{\rho},

    where :math:`f(t)` is the force density at time :math:`t`, :math:`\eta` is
    the dynamic relaxation damping constant and :math:`\rho` is the density.
    """

    def __init__(self, damping, *args, **kwargs):
        """
        Create an :class:`EulerCromerCL` integrator object.

        :arg float damping: The dynamic relaxation damping constant with units
            [kg/(m^3 s)]

        :returns: An :class:`EulerCromerCL` object
        """
        super().__init__(*args, **kwargs)
        self.damping = damping

    def __call__(self, displacement_bc_magnitude, force_bc_magnitude):
        """
        Conduct one iteration of the integrator.

        :arg float displacement_bc_magnitude: the magnitude applied to the
             displacement boundary conditions for the current time-step.
        :arg float force_bc_magnitude: the magnitude applied to the force
            boundary conditions for the current time-step.
        """
        self._bond_force(
            self.u_d, self.force_d, self.body_force_d, self.r0_d, self.vols_d,
            self.nlist_d, self.force_bc_types_d, self.force_bc_values_d,
            self.stiffness_corrections_d, self.bond_types_d, self.regimes_d,
            self.plus_cs_d, self.local_mem_x, self.local_mem_y,
            self.local_mem_z, self.bond_stiffness_d, self.critical_stretch_d,
            force_bc_magnitude, self.nregimes)

        self._update_displacement(
            self.force_d, self.u_d, self.ud_d, self.udd_d, self.bc_types_d,
            self.bc_values_d, self.densities_d, displacement_bc_magnitude,
            self.damping, self.dt)

    def _build_special(self):
        """Build OpenCL kernels special to the Euler integrator."""
        if self.densities is None:
            raise ValueError(
                "densities must be supplied when using EulerCromerCL "
                "integrator (got {}). This integrator is dynamic "
                " and requires the density or is_density argument to be "
                "supplied to :class:Model, alternatively, use a static "
                " integrator, such as EulerCL.".format(type(self.densities)))
        else:
            self.densities_d = cl.Buffer(
                self.context, mf.READ_ONLY | mf.COPY_HOST_PTR,
                hostbuf=self.densities)

        kernel_source = open(
            pathlib.Path(__file__).parent.absolute() /
            "cl/euler_cromer.cl").read()

        # Build kernels
        self.euler_cromer = cl.Program(
            self.context, kernel_source).build()
        self.update_displacement_kernel = self.euler_cromer.update_displacement

    def _create_special_buffers(self):
        """Create buffers special to the Euler integrator."""
        # There are none

    def _update_displacement(
            self, force_d, u_d, ud_d, udd_d, bc_types_d, bc_values_d,
            densities_d, displacement_bc_magnitude, damping, dt):
        """Update displacements."""
        queue = self.queue
        # Call kernel
        self.update_displacement_kernel(
                self.queue, (self.degrees_freedom * self.nnodes,), None,
                force_d, u_d, ud_d, udd_d, bc_types_d, bc_values_d,
                densities_d, np.float64(displacement_bc_magnitude),
                np.float64(damping), np.float64(dt)
                )
        queue.finish()
        return u_d


class VelocityVerletCL(Integrator):
    r"""
    Velocity-Verlet integrator for OpenCL.

    The Velocity-Verlet method is a second-order numerical integration method.
    The integration is given by,

    .. math::
        \dot{u}(t + \frac{\delta t}{2}) = \dot{u}(t) +
        \frac{\delta t}{2}\ddot{u}(t),
    .. math::
        u(t + \delta t) = u(t) + \delta t \dot{u}(t)
                            + \frac{\delta t}{2} \ddot{u}(t),
    .. math::
        \dot{u}(t + \delta t) = \dot{u}(t + \frac{\delta t}{2})
                            + \frac{\delta t}{2} \ddot{u}(t + \delta t),

    where :math:`u(t)` is the displacement at time :math:`t`,
    :math:`\dot{u}(t)` is the velocity at time :math:`t`, :math:`\ddot{u}(t)`
    is the acceleration at time :math:`t` and :math:`\delta t` is the time
    step.

    A dynamic relaxation damping term :math:`\eta \dot{u}(t)` is added to the
    equation of motion so that the solution to quickly converges to a steady
    state solution in quasi-static problems. Given the displacement vectors of
    each node at time step t, and half-step velocity vectors of each node at
    time step :math:`t + \frac{\delta t}{2}`, the acceleration at time step
    :math:`t + \delta t` is given by the equation of motion,

    .. math::
        \ddot{u}(t + \delta t) = \frac{f(t + \delta t)
                             - \eta \dot{u}(t + \frac{\delta t}{2})}{\rho},

    where :math:`f(t)` is the force density at time :math:`t`, :math:`\eta`
    is the dynamic relaxation damping constant and :math:`\rho` is the density.
    """

    def __init__(self, damping, *args, **kwargs):
        """
        Create an :class:`VelocityVerletCL` integrator object.

        :arg float damping: The dynamic relaxation damping constant with units
            [kg/(m^3 s)]

        :returns: A :class:`VelocityVerletCL` object
        """
        super().__init__(*args, **kwargs)
        self.damping = damping

    def __call__(self, displacement_bc_magnitude, force_bc_magnitude):
        """
        Conduct one iteration of the integrator.

        :arg float displacement_bc_magnitude: the magnitude applied to the
             displacement boundary conditions for the current time-step.
        :arg float force_bc_magnitude: the magnitude applied to the force
            boundary conditions for the current time-step.
        """
        self._bond_force(
            self.u_d, self.force_d, self.body_force_d, self.r0_d, self.vols_d,
            self.nlist_d, self.force_bc_types_d, self.force_bc_values_d,
            self.stiffness_corrections_d, self.bond_types_d, self.regimes_d,
            self.plus_cs_d, self.local_mem_x, self.local_mem_y,
            self.local_mem_z, self.bond_stiffness_d, self.critical_stretch_d,
            force_bc_magnitude, self.nregimes)

        self._update_displacement(
            self.force_d, self.u_d, self.ud_d, self.udd_d, self.bc_types_d,
            self.bc_values_d, self.densities_d, displacement_bc_magnitude,
            self.damping, self.dt)

    def _build_special(self):
        """Build OpenCL kernels special to the Euler integrator."""
        if self.densities is None:
            raise ValueError(
                "densities must be supplied when using VelocityVerletCL "
                "integrator (got {}). This integrator is dynamic "
                " and requires the density or is_density argument to be "
                "supplied to :class:Model, alternatively, use a static "
                " integrator, such as EulerCL.".format(type(self.densities)))
        else:
            self.densities_d = cl.Buffer(
                self.context, mf.READ_ONLY | mf.COPY_HOST_PTR,
                hostbuf=self.densities)

        kernel_source = open(
            pathlib.Path(__file__).parent.absolute() /
            "cl/velocity_verlet.cl").read()

        # Build kernels
        self.euler_cromer = cl.Program(
            self.context, kernel_source).build()
        self.update_displacement_kernel = self.euler_cromer.update_displacement
        self.partial_update_displacement_kernel = (
            self.euler_cromer.update_displacement)

    def _create_special_buffers(self):
        """Create buffers special to the Euler integrator."""
        # There are none

    def _update_displacement(
            self, force_d, u_d, ud_d, udd_d, bc_types_d, bc_values_d,
            densities_d, displacement_bc_magnitude, damping, dt):
        """Update displacements."""
        queue = self.queue
        # Call kernel
        self.update_displacement_kernel(
                self.queue, (self.degrees_freedom * self.nnodes,), None,
                force_d, u_d, ud_d, udd_d, bc_types_d, bc_values_d,
                densities_d, np.float64(displacement_bc_magnitude),
                np.float64(damping), np.float64(dt)
                )
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

"""Peridynamics model using Ben's Optimised OpenCL kernels."""
from .model import Model
from .cl_ben import double_fp_support, get_context, kernel_source
import numpy as np
import pyopencl as cl
from pyopencl import mem_flags as mf
import pathlib
from tqdm import trange
import sys


class ModelCLBen(Model):
    """
    A peridynamics model using Ben's optimised OpenCL code.

    This class allows users to define an composite peridynamics system with
    any number of materials and damage laws from parameters and a set of
    initial conditions (coordinates, connectivity, material_types and
    stiffness_corrections).
    """

    def __init__(self, *args, density=None, bond_type=None,
                 material_types=None, stiffness_corrections=None,
                 precise_stiffness_correction=None, dt=None, write_path=None,
                 context=None, **kwargs):
        """
        Create a :class:`ModelCLBen` object.

        :arg float density: Density of the bulk material in kg/m^3.
        :arg bond_stiffness: An (n_regimes, n_materials) array of bond
            stiffness values, each corresponding to a material and a regime.
        :type bond_stiffness: list or :class `numpy.ndarray`:
        :arg critical_stretch: An (n_regimes, n_materials) array of critical
            stretch values, each corresponding to a material and a regime.
        :type critical_stretch: list or :class `numpy.ndarray`:
        :arg method bond_type: A method which outputs the material type,
            an integer value, of the bond.
        :arg connectivity: The initial connectivity for the model. A tuple
            of a neighbour list and the number of neighbours for each node. If
            `None` the connectivity at the time of construction of the
            :class:`Model` object will be used. Default `None`.
        :type connectivity: tuple(:class:`numpy.ndarray`,
            :class:`numpy.ndarray`)
        :arg material_types: The bond material_types for the model.
            If `None` the material_types at the time of construction of the
            :class:`Model` object will be used. Default `None`.
        :type material_types: :class:`numpy.ndarray`
        :arg stiffness_corrections: The stiffness_corrections for
            the model. If `None` the stiffness_corrections at the time
            of construction of the :class:`Model` object will be used. Default
            `None`.
        :type stiffness_corrections:
        :arg bool transfinite: Cartesian cubic (tensor grid) mesh (1) or
            tetra-hedral grid (default, 0).
        :arg bool precise_stiffness_correction: Boolean for stiffness
            correction factors calculated using mesh element volumes (default
            'precise', 1) or average nodal volume of a transfinite mesh (0). If
            `None`, then no stiffness correction factors are provided.
        :arg float max_reaction: The maximum total load applied to the loaded
            nodes.
        :arg int build_load: The number of steps to apply the max reaction
            force to the loaded nodes at a linear rate.
        :arg int build_displacement: The number of steps to apply the build up
            for the displacement.
        :arg float max_displacement: The maximum displacement applied to the
            loaded nodes.
        :arg write_path: The path where the stiffness_corrections,
            material_types and connectivity should be written.
        :type write_path: path-like or str

        :returns: A new :class:`Model` object.
        :rtype: Model
        """
        super().__init__(*args, **kwargs)

        # If no write path was provided use the current directory, otherwise
        # ensure write_path is a Path object.
        if write_path is None:
            write_path = pathlib.Path()
        else:
            write_path = pathlib.Path(write_path)
        self.degrees_freedom = 3
        self.precise_stiffness_correction = precise_stiffness_correction
        self.density = density
        self.dt = dt

        if stiffness_corrections is None:
            # Calculate stiffness correction factors and write to file
            self.stiffness_corrections = \
                self._set_stiffness_corrections(
                    self.horizon, self.initial_connectivity,
                    precise_stiffness_correction, write_path)
        elif type(stiffness_corrections) == np.ndarray:
            if np.shape(stiffness_corrections) != (
                    self.nnodes, self.max_neighbours):
                raise ValueError("stiffness_corrections must be of \
                                 shape (nnodes, max_neighbours)")
            else:
                self.stiffness_corrections = stiffness_corrections
        else:
            raise TypeError(
                "stiffness_corrections must be a numpy.ndarray or None")

        # Create dummy boundary conditions function is none is provided
        if bond_type is None:
            def bond_type(x, y):
                return 0

        if material_types is None:
            # Calculate material types and write to file
            self.material_types = self._set_material_types(
                self.initial_connectivity, bond_type, write_path)
        elif type(material_types) == np.ndarray:
            if np.shape(material_types) != (self.nnodes, self.max_neighbours):
                raise ValueError("material_types must be of shape\
                                 (nnodes, max_neighbours)")
            else:
                self.material_types = material_types
        else:
            raise TypeError("stiffness_corrections must be an \
                            numpy.ndarray or None")

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

        # JIT Compiler command line arguments
        SEP = " "
        options_string = (
            "-cl-fast-relaxed-math" + SEP
            + "-DPD_DPN_NODE_NO=" + str(self.degrees_freedom * self.nnodes)
            + SEP
            + "-DPD_NODE_NO=" + str(self.nnodes) + SEP
            + "-DMAX_HORIZON_LENGTH=" + str(self.max_neighbours) + SEP
            + "-DPD_DT=" + str(self.dt) + SEP
            + "-DPD_REGIME_NO=" + str(self.n_regimes) + SEP)

        # Build kernels
        self.program = cl.Program(
            self.context, kernel_source).build([options_string])
        self.queue = cl.CommandQueue(self.context)

        self.bond_force_kernel = self.program.bond_force_new
        self.update_displacement_kernel = self.program.update_displacement
        self.damage_kernel = self.program.damage
        self.damage_new_kernel = self.program.damage_new

    def _damage(self, nlist_d, family_d, damage_d, local_mem):
        """Calculate bond damage."""
        queue = self.queue

        # Call kernel
        self.damage_kernel(
            self.queue, (self.nnodes * self.max_horizon_length,),
            (self.max_horizon_length,), nlist_d, family_d, damage_d, local_mem)
        queue.finish()

    def _damage_new(self, n_neigh_d, family_d, damage_d):
        """Calculate bond damage."""
        queue = self.queue

        # Call kernel
        self.damage_new_kernel(queue, (self.nnodes,), None, n_neigh_d,
                               family_d, damage_d)
        queue.finish()

    def _bond_force(
            self, u_d, ud_d, r0_d, vols_d, nlist_d, n_neigh_d,
            stiffness_corrections_d, material_types_d, regimes_d,
            bond_stiffness_d, critical_stretch_d, plus_cs_d, force_bc_types_d,
            force_bc_values_d, local_mem_x, local_mem_y, local_mem_z,
            force_load_scale):
        """Calculate the force due to bonds acting on each node."""
        queue = self.queue
        # Call kernel
        self.bond_force_kernel(
                queue, (self.nnodes * self.max_neighbours,),
                (self.max_neighbours,), u_d, ud_d, r0_d, vols_d, nlist_d,
                n_neigh_d, stiffness_corrections_d, material_types_d,
                regimes_d, bond_stiffness_d, critical_stretch_d, plus_cs_d,
                force_bc_types_d, force_bc_values_d, local_mem_x, local_mem_y,
                local_mem_z, force_load_scale)
        queue.finish()
        return ud_d, nlist_d, n_neigh_d, regimes_d

    def _update_displacement(
            self, ud_d, u_d, bc_types_d, bc_values_d, displacement_load_scale):
        """Update displacements."""
        queue = self.queue
        # Call kernel
        self.update_displacement_kernel(
                self.queue, (self.degrees_freedom * self.nnodes,), None,
                ud_d, u_d, bc_types_d, bc_values_d, displacement_load_scale)
        queue.finish()
        return u_d

    def write_array(self, write_path, array):
        """
        Write a numpy array to a vtk file.

        :arg write_path: The path where the vtk files should be
            written.
        :type write_path: path-like or str
        :arg numpy.ndarray array: The array to be written.

        :return: None
        :rtype: NoneType
        """
        f = open(write_path, "w")
        f.write("# vtk DataFile Version 2.0\n")
        f.write("ASCII\n")
        f.write("\n")
        f.write("DATASET ARRAY\n")
        if type(array[0][0]) is (np.float64 or float):
            for i in range(0, np.shape(array)[0]):
                tmp = array[i]
                for j in range(0, len(tmp)):
                    f.write("{:f} ".format(tmp[j]))
                f.write("\n")
            f.close()
        elif type(array[0][0]) is (np.intc or int):
            for i in range(0, np.shape(array)[0]):
                tmp = array[i]
                for j in range(0, len(tmp)):
                    f.write("{:d} ".format(np.intc(tmp[j])))
                f.write("\n")
            f.close()
        else:
            raise ValueError(
                'values type not recognised, could not write to file.\
                Type must be float, numpy.float64, int or numpy.intc')

    def _set_material_types(self, connectivity, bond_type, write_path):
        """
        Build material_types array.

        Builds a (`nnodes`, `max_neighbours`) array of material types for each
        bond for each node.

        :arg connectivity: The initial connectivity for the simulation. A tuple
            of a neighbour list and the number of neighbours for each node. If
            `None` the connectivity at the time of construction of the
            :class:`Model` object will be used. Default `None`.
        :type connectivity: tuple(:class:`numpy.ndarray`,
            :class:`numpy.ndarray`)
        :arg bond_type: A function that returns an integer value depending on
            the material type.
        :arg write_path: The path where the vtk files should be written.
        :type write_path: path-like or str

        :returns: A (`nnodes`, `max_neighbours`) array of the material type
            of each bond for each node, which are used
            to index into the bond_stiffness and critical_stretch arrays.
        :rtype: :class:`numpy.ndarray`
        """
        nlist, n_neigh = connectivity
        material_types = np.zeros(
            (self.nnodes, self.max_neighbours), dtype=np.intc)
        print(material_types.shape, nlist.shape, self.coords.shape)
        for i in range(self.nnodes):
            for neigh in range(n_neigh[i]):
                j = nlist[i][neigh]
                material_types[i][neigh] = bond_type(
                    self.coords[i, :], self.coords[j, :])
        material_types = material_types.astype(np.intc)
        self.write_array(write_path/"material_types", material_types)
        return material_types

    def _set_stiffness_corrections(
            self, horizon, connectivity,
            precise_stiffness_correction, write_path):
        """
        Build a list of stiffness correction factors.

        Stiffness correction factors reduce the peridynamics surface softening
        effect for 2D/3D problem and writes to file. The 'volume method'
        proposed in Chapter 2 in Bobaru F, Foster JT, Geubelle PH, Silling SA
        (2017) Handbook of peridynamic modeling (p51 – 52) is used here.

        :arg float horizon: The horizon distance.
        :arg connectivity: The initial connectivity for the simulation. A tuple
            of a neighbour list and the number of neighbours for each node. If
            `None` the connectivity at the time of construction of the
            :class:`Model` object will be used. Default `None`.
        :type connectivity: tuple(:class:`numpy.ndarray`,
            :class:`numpy.ndarray`)
        :arg precise_stiffness_correction int: A switch variable. Set to 1:
            Stiffness corrections are calculated more accurately using
            actual nodal volumes. Set to 0: Stiffness corrections are calculate
            using an average nodal volume. Set to None: All stiffness
            corrections are set to 1.0, i.e. no stiffness correction is
            applied.
        :arg write_path: The path where the vtk files should be written.
        :type write_path: path-like or str

        :returns: A (`nnodes`, `max_neighbours`) array of the stiffness
            correction factor of each bond for each node.
        :rtype: :class:`numpy.ndarray`
        """
        nlist, n_neigh = connectivity
        stiffness_corrections = np.ones((self.nnodes, self.max_neighbours))
        family_volumes = np.zeros(self.nnodes)
        for i in range(0, self.nnodes):
            tmp = 0.0
            neighbour_list = nlist[i][:self.family[i]]
            for j in range(self.family[i]):
                tmp += self.volume[neighbour_list[j]]
            family_volumes[i] = tmp

        if self.dimensions == 2:
            family_volume_bulk = np.pi*np.power(horizon, 2) * 0.001
        elif self.dimensions == 3:
            family_volume_bulk = (4./3)*np.pi*np.power(horizon, 3)

        if precise_stiffness_correction == 1:
            for i in range(0, self.nnodes):
                family_volume_i = family_volumes[i]
                for neigh in range(n_neigh[i]):
                    family_volume_j = family_volumes[nlist[i][neigh]]
                    stiffness_correction_factor = 2. * family_volume_bulk / (
                        family_volume_i + family_volume_j)
                    stiffness_corrections[i][neigh] = (
                        stiffness_correction_factor)

        elif precise_stiffness_correction == 0:
            average_node_volume = self.volume_total / self.nnodes
            for i in range(0, self.nnodes):
                nnodes_i_family = n_neigh[i]
                nodei_family_volume = nnodes_i_family * average_node_volume
                for neigh in nnodes_i_family:
                    j = nlist[i][neigh]
                    nnodes_j_family = n_neigh[j]
                    nodej_family_volume = nnodes_j_family * average_node_volume
                    stiffness_correction_factor = 2. * family_volume_bulk / (
                        nodej_family_volume + nodei_family_volume)
                    stiffness_corrections[i][neigh] = (
                        stiffness_correction_factor)

        elif precise_stiffness_correction is None:
            pass
        else:
            raise ValueError('precise_stiffness_correction can \
                             only take values 0 or 1 or None')
        self.write_array(
            write_path/"stiffness_corrections", stiffness_corrections)
        return stiffness_corrections

    def _set_plus_cs(self, bond_stiffness, critical_stretch, n_regimes,
                     n_materials):
        """
        Calculate `+ c`s for the damage models.

        Calculates the `+ c`s (c.f. `y = mx + c`) for the n-linear
        damage-model, where n is n_regimes, e.g. linear, bi-linear, tri-linear,
        etc. from the bond_stiffness and critical_stretch values provided.

        :arg bond_stiffness: An (n_regimes, n_materials) array of bond
            stiffness values, each corresponding to a material and a regime.
        :type bond_stiffness: list or :class `numpy.ndarray`:
        :arg critical_stretch: An (n_regimes, n_materials) array of critical
            stretch values, each corresponding to a material and a regime.
        :type critical_stretch: list or :class `numpy.ndarray`:
        :arg int n_regimes: The number of `regimes` in the damage model. e.g.
            linear has n_regimes = 1, bi-linear has n_regimes = 2, etc.
        :arg n_materials: The number of materials in the model.

        :returns: A (`n_regimes`, `n_materials`) array of the `+cs` for each
            linear part of the bond damage models for each material.
        :rtype: :class:`numpy.ndarray`
        """
        # For initial elastic regime, the bond force density at 0 stretch is 0
        c0 = 0.0
        c_prev = c0
        plus_cs = [c0]
        if n_regimes != 1:
            # infer the number of materials in the model from the array shape
            # TODO: generalise for n-material types.
            for i in range(n_regimes - 1):
                c_i = c_prev + bond_stiffness[i - 1] * critical_stretch[i - 1]\
                    - bond_stiffness[i] * critical_stretch[i - 1]
                plus_cs.append[c_i]
                c_prev = c_i
        assert len(plus_cs) == n_regimes
        return plus_cs

    def _increment_load(self, build_load, step):
        """
        Increment and update the force boundary conditions.

        :arg float build_load: The inverse of the number of steps required to
            build up to full external force loading.
        :arg int step: The current time-step of the simulation.

        :returns: The force_load_scale between [0.0, 1.0], a scale applied to
            the force boundary conditions.
        :rtype: :class:`numpy.float64`
        """
        # Increase load in linear increments
        if not (build_load is None):
            force_load_scale = np.float64(min(1.0, build_load * step))
        else:
            force_load_scale = np.float64(0.0)
        return force_load_scale

    def _increment_displacement(self, coefficients, build_time, step, ease_off,
                                displacement_rate, build_displacement,
                                final_displacement):
        """
        Increment the displacement boundary condition values.

        According to a 5th order polynomial/ linear displacement-time curve
        for which initial acceleration is 0.

        :arg tuple coefficients: Tuple containing the 3 free coefficients
            of the 5th order polynomial.
        :arg int build_time: The number of time steps over which the
            applied displacement-time curve is not linear.
        :arg int step: The current time-step of the simulation.
        :arg int ease_off: A boolean-like variable which is 0 if the
            displacement-rate hasn't started decreasing yet. Equal to the step
            at which the displacement rate starts decreasing once it does so.
        :arg float displacement_rate: The displacement rate in [m] per step
            during the linear phase of the displacement-time graph.
        :arg float build_displacement: The displacement in [m] over which the
            displacement-time graph is the smooth 5th order polynomial.
        :arg float final_displacement: The final applied displacement in [m].

        :returns: The displacement_load_scale between [0.0, 1.0], a scale
            applied to the displacement boundary conditions.
        :rtype: np.float64
        :returns: ease_off
        :rtype: int
        """
        if not ((displacement_rate is None) or (build_displacement is None)
                or (final_displacement is None)):
            # Calculate the scale applied to the displacements
            displacement_scale, ease_off = _calc_displacement_scale(
                coefficients, final_displacement, build_time,
                displacement_rate, step, build_displacement, ease_off)
            if displacement_scale != 0.0:
                # update the host force load scale
                displacement_load_scale = np.float64(displacement_scale)
        # No user specified build up parameters case
        elif not (displacement_rate is None):
            # update the host force load scale
            displacement_load_scale = np.float64(1.0)
        return displacement_load_scale, ease_off

    def simulate(self, steps, u=None, ud=None, connectivity=None,
                 regimes=None, bond_stiffness=None, critical_stretch=None,
                 is_boundary=None, is_forces_boundary=None, is_tip=None,
                 displacement_rate=None, build_displacement=None,
                 final_displacement=None, build_load=None,
                 max_reaction=None, first_step=1, write=None,
                 write_path=None):
        """
        Simulate the peridynamics model.

        :arg int steps: The number of simulation steps to conduct.
        :arg u: The initial displacements for the simulation. If `None` the
            displacements will be initialised to zero. Default `None`.
        :type u: :class:`numpy.ndarray`
        :arg ud: The initial velocities for the simulation. If `None` the
            velocities will be initialised to zero. Default `None`.
        :type ud: :class:`numpy.ndarray`
        :arg connectivity: The initial connectivity for the simulation. A tuple
            of a neighbour list and the number of neighbours for each node. If
            `None` the connectivity at the time of construction of the
            :class:`Model` object will be used. Default `None`.
        :type connectivity: tuple(:class:`numpy.ndarray`,
            :class:`numpy.ndarray`)
        :arg regimes: The initial regimes for the simulation. A
            (`nodes`, `max_neighbours`) array of type
            :class:`numpy.ndarray` of the regimes of the bonds
            of a neighbour list and the number of neighbours for each node.
        :type connectivity: tuple(:class:`numpy.ndarray`,
            :class:`numpy.ndarray`)
        :arg is_boundary: A function to determine if a node is on the boundary
            for a displacement boundary condition, and if it is, which
            direction the boundary conditions are applied
            (positive or negative cartesian direction). It has the form
            is_boundary(:class:`numpy.ndarray`). The argument is the initial
            coordinates of a particle being simulated. `is_boundary` returns a
            (3) list of the boundary types in each cartesian direction.
            A boundary type with an int value of 2 if the particle is not on a
            displacement controlled boundary, a value of 1 if is is on a
            boundary and loaded in the positive cartesian direction, and a
            value of -1 if it is on the boundary and loaded in the negative
            direction, and a value of 0 if it is not loaded.
        :type is_boundary: function
        :arg is_forces_boundary: As 'is_boundary' but applying to force
            boundary conditions as opposed to displacement boundary conditions.
        :type is_forces_boundary: function
        :arg is_tip: A function to determine if a node is to be measured for
            its reaction force or displacement over time, and if it is, which
            direction the measurements are made
            (positive or negative cartesian direction). It has the form
            is_tip(:class:`numpy.ndarray`). The argument is the initial
            coordinates of a particle being simulated. `is_tip` returns a
            (3) list of the measurement types in each cartesian direction.
            A boundary type with an int value of 2 if the particle is not on
            the `tip` to be measured, a value of 1 if is is on the `tip` and
            measured in the positive cartesian direction, and a value of -1 if
            it is on the `tip` and measured in the negative direction.
        :type is_tip: function
        :arg float displacement_rate: The displacement rate in [m] per step
            during the linear phase of the displacement-time graph, and the
            maximum displacement rate of any part of the simulation.
        :arg float build_displacement: The displacement in [m] over which the
            displacement-time graph is the smooth 5th order polynomial.
        :arg float final_displacement: The final applied displacement in [m].
        :arg float build_load: The inverse of the number of steps required to
            build up to full external force loading.
        :arg float max_reaction: The maximum total load applied to the loaded
            nodes.
        :arg int first_step: The starting step number. This is useful when
            restarting a simulation, especially if `boundary_function` depends
            on the absolute step number.
        :arg int write: The frequency, in number of steps, to write the system
            to a mesh file by calling :meth:`Model.write_mesh`. If `None` then
            no output is written. Default `None`.
        :arg write_path: The path where the periodic mesh files should be
            written.
        :type write_path: path-like or str

        :returns: A tuple of the final displacements (`u`), the final
            velocities (`ud`), damage, connectivity, a (steps) list of the
            total sum of all damage over the time steps, a (steps, 3) array of
            the tip displacements over the time-steps and a (steps, 3) array of
            the tip resultant force over the time-steps.
        :rtype: tuple(:class:`numpy.ndarray`, :class:`numpy.ndarray`,
                      :class:`numpy.ndarray`,
            tuple(:class:`numpy.ndarray`, :class:`numpy.ndarray`))
        """
        (nlist,
         n_neigh,
         regimes,
         bond_stiffness,
         critical_stretch,
         plus_cs,
         u,
         ud,
         damage,
         bc_types,
         bc_values,
         force_bc_types,
         force_bc_values,
         tip_types,
         write_path) = self._simulate_initialise(
             is_boundary, is_forces_boundary, is_tip, displacement_rate,
             max_reaction, u, ud, connectivity, regimes, bond_stiffness,
             critical_stretch, write_path)

        # Calculate no. of time steps that applied BCs are in the build phase
        if not ((displacement_rate is None)
                or (build_displacement is None)
                or (final_displacement is None)):
            build_time, coefficients = _calc_build_time(
                build_displacement, displacement_rate, steps)
        else:
            build_time, coefficients = None, None

        # Local memory containers for Bond forces
        local_mem_x = cl.LocalMemory(
            np.dtype(np.float64).itemsize * self.max_neighbours)
        local_mem_y = cl.LocalMemory(
            np.dtype(np.float64).itemsize * self.max_neighbours)
        local_mem_z = cl.LocalMemory(
            np.dtype(np.float64).itemsize * self.max_neighbours)

        # For applying force in incriments
        force_load_scale = np.float64(0.0)
        # For applying displacement in incriments
        displacement_load_scale = np.float64(0.0)

        # Build OpenCL data structures

        # Read only
        r0_d = cl.Buffer(
            self.context, mf.READ_ONLY | mf.COPY_HOST_PTR,
            hostbuf=np.ascontiguousarray(self.coords, dtype=np.float64))
        bc_types_d = cl.Buffer(
            self.context, mf.READ_ONLY | mf.COPY_HOST_PTR,
            hostbuf=bc_types)
        bc_values_d = cl.Buffer(
            self.context, mf.READ_ONLY | mf.COPY_HOST_PTR,
            hostbuf=bc_values)
        force_bc_types_d = cl.Buffer(
            self.context, mf.READ_ONLY | mf.COPY_HOST_PTR,
            hostbuf=force_bc_types)
        force_bc_values_d = cl.Buffer(
            self.context, mf.READ_ONLY | mf.COPY_HOST_PTR,
            hostbuf=force_bc_values)
        vols_d = cl.Buffer(
            self.context, mf.READ_ONLY | mf.COPY_HOST_PTR,
            hostbuf=self.volume)
        stiffness_corrections_d = cl.Buffer(
            self.context, mf.READ_ONLY | mf.COPY_HOST_PTR,
            hostbuf=np.ascontiguousarray(
                self.stiffness_corrections, dtype=np.float64))
        material_types_d = cl.Buffer(
            self.context, mf.READ_ONLY | mf.COPY_HOST_PTR,
            hostbuf=np.ascontiguousarray(
                self.material_types, dtype=np.float64))
        bond_stiffness_d = cl.Buffer(
            self.context, mf.READ_ONLY | mf.COPY_HOST_PTR,
            hostbuf=np.ascontiguousarray(
                bond_stiffness, dtype=np.float64))
        critical_stretch_d = cl.Buffer(
            self.context, mf.READ_ONLY | mf.COPY_HOST_PTR,
            hostbuf=np.ascontiguousarray(critical_stretch, dtype=np.float64))
        plus_cs_d = cl.Buffer(
            self.context, mf.READ_ONLY | mf.COPY_HOST_PTR,
            hostbuf=np.ascontiguousarray(plus_cs, dtype=np.float64))
        family_d = cl.Buffer(
            self.context, mf.READ_ONLY | mf.COPY_HOST_PTR,
            hostbuf=self.family)
        # Read and write
        regimes_d = cl.Buffer(
            self.context, mf.READ_WRITE | mf.COPY_HOST_PTR,
            hostbuf=np.ascontiguousarray(regimes, dtype=np.float64))
        nlist_d = cl.Buffer(
            self.context, mf.READ_WRITE | mf.COPY_HOST_PTR,
            hostbuf=nlist)
        n_neigh_d = cl.Buffer(
            self.context, mf.READ_WRITE | mf.COPY_HOST_PTR,
            hostbuf=n_neigh)
        u_d = cl.Buffer(
            self.context, mf.READ_WRITE, np.empty(
                (self.nnodes, self.degrees_freedom), dtype=np.float64).nbytes)
        ud_d = cl.Buffer(
            self.context, mf.READ_WRITE, np.empty(
                (self.nnodes, self.degrees_freedom), dtype=np.float64).nbytes)

        # Write only
        damage_d = cl.Buffer(
            self.context, mf.WRITE_ONLY, damage.nbytes)
        # Initialize kernel parameters
        # TODO: no sign of this in Jim's code, is it necessary?
# =============================================================================
#         self.bond_force_kernel.set_scalar_arg_dtypes(
#             [None, None, None, None, None,
#              None, None, None, None, None,
#              None, None, None, None, None,
#              None, None, None])
#         # Initialize kernel parameters
#         self.update_displacement_kernel.set_scalar_arg_dtypes(
#             [None, None, None, None, None])
#         self.damage_kernel.set_scalar_arg_dtypes(
#             [None, None, None, None])
#         self.damage_kernel.set_scalar_arg_dtypes(
#             [None, None, None])
# =============================================================================

        # Container for plotting data
        damage_sum_data = []
        tip_displacement_data = []
        tip_force_data = []

        # Ease off displacement loading switch
        ease_off = 0
        for step in trange(first_step, first_step+steps,
                           desc="Simulation Progress", unit="steps"):

            # Update displacements
            u_d = self._update_displacement(
                ud_d, u_d, bc_types_d, bc_values_d, displacement_load_scale)

            # Calculate the force due to bonds on each node,
            # and update connectivity
            ud_d, nlist_d, n_neigh_d, regimes_d = self._bond_force(
                u_d, ud_d, r0_d, vols_d, nlist_d, n_neigh_d,
                stiffness_corrections_d, material_types_d, regimes_d,
                bond_stiffness_d, critical_stretch_d, plus_cs_d,
                force_bc_types_d, force_bc_values_d, local_mem_x, local_mem_y,
                local_mem_z, force_load_scale)

            if write:
                if step % write == 0:
                    # Calculate the damage
                    self._damage_new(n_neigh_d, family_d, damage_d)

                    cl.enqueue_copy(self.queue, damage, damage_d)
                    cl.enqueue_copy(self.queue, u, u_d)
                    cl.enqueue_copy(self.queue, ud, ud_d)

                    self.write_mesh(write_path/f"U_{step}.vtk", damage, u)

                    tip_displacement = 0
                    tip_shear_force = 0
                    tmp = 0
                    for i in range(self.nnodes):
                        if tip_types[i] == 1:
                            tmp += 1
                            tip_displacement += u[i][2]
                            tip_shear_force += ud[i][2]
                    if tmp != 0:
                        tip_displacement /= tmp
                    else:
                        tip_displacement = None

                    tip_displacement_data.append(tip_displacement)
                    tip_force_data.append(tip_shear_force)
                    damage_sum = np.sum(damage)
                    damage_sum_data.append(damage_sum)
                    if damage_sum > 0.02*self.nnodes:
                        print('Warning: over 2% of bonds have broken! \
                              -- PERIDYNAMICS SIMULATION CONTINUING')
                    elif damage_sum > 0.7*self.nnodes:
                        print('Warning: over 7% of bonds have broken! \
                              -- PERIDYNAMICS SIMULATION STOPPING')
                        break

            # Increase external forces in linear incremenets
            if force_load_scale != 1.0:
                force_load_scale = self._increment_load(build_load, step)
            # Increase displacement in 5th order polynomial increments
            displacement_load_scale, ease_off = self._increment_displacement(
                coefficients, build_time, step, ease_off, displacement_rate,
                build_displacement, final_displacement)

        return (u, ud, damage, (nlist, n_neigh), damage_sum_data,
                tip_displacement_data, tip_force_data)

    def _simulate_initialise(
            self, is_boundary, is_forces_boundary, is_tip, displacement_rate,
            regimes, max_reaction, u, ud, connectivity, bond_stiffness,
            critical_stretch, write_path):
        """
        Initialise simulation variables.

        :arg is_boundary: A function to determine if a node is on the boundary
            for a displacement boundary condition, and if it is, which
            direction the boundary conditions are applied
            (positive or negative cartesian direction). It has the form
            is_boundary(:class:`numpy.ndarray`). The argument is the initial
            coordinates of a particle being simulated. `is_boundary` returns a
            (3) list of the boundary types in each cartesian direction.
            A boundary type with an int value of 2 if the particle is not on a
            displacement controlled boundary, a value of 1 if is is on a
            boundary and loaded in the positive cartesian direction, and a
            value of -1 if it is on the boundary and loaded in the negative
            direction, and a value of 0 if it is not loaded.
        :type is_boundary: function
        :arg is_forces_boundary: As 'is_boundary' but applying to force
            boundary conditions as opposed to displacement boundary conditions.
        :type is_forces_boundary: function
        :arg is_tip: A function to determine if a node is to be measured for
            its reaction force or displacement over time, and if it is, which
            direction the measurements are made
            (positive or negative cartesian direction). It has the form
            is_tip(:class:`numpy.ndarray`). The argument is the initial
            coordinates of a particle being simulated. `is_tip` returns a
            (3) list of the measurement types in each cartesian direction.
            A boundary type with an int value of 2 if the particle is not on
            the `tip` to be measured, a value of 1 if is is on the `tip` and
            measured in the positive cartesian direction, and a value of -1 if
            it is on the `tip` and measured in the negative direction.
        :type is_tip: function
        :arg float displacement_rate: The displacement rate in [m] per step
            during the linear phase of the displacement-time graph, and the
            maximum displacement rate of any part of the simulation.
        :arg float max_reaction: The maximum total load applied to the loaded
            nodes.
        :arg u: The initial displacements for the simulation. If `None` the
            displacements will be initialised to zero. Default `None`.
        :type u: :class:`numpy.ndarray`
        :arg u: The initial displacements for the simulation. If `None` the
            displacements will be initialised to zero. Default `None`.
        :type u: :class:`numpy.ndarray`
        :arg ud: The initial velocities for the simulation. If `None` the
            velocities will be initialised to zero. Default `None`.
        :type ud: :class:`numpy.ndarray`
        :arg connectivity: The initial connectivity for the simulation. A tuple
            of a neighbour list and the number of neighbours for each node. If
            `None` the connectivity at the time of construction of the
            :class:`Model` object will be used. Default `None`.
        :type connectivity: tuple(:class:`numpy.ndarray`,
            :class:`numpy.ndarray`)
        :arg bond_stiffness: An (n_regimes, n_materials) array of bond
            stiffness values, each corresponding to a material and a regime.
        :type bond_stiffness: list or :class `numpy.ndarray`:
        :arg critical_stretch: An (n_regimes, n_materials) array of critical
            stretch values, each corresponding to a material and a regime.
        :type critical_stretch: list or :class `numpy.ndarray`:
        :arg write_path: The path where the periodic mesh files should be
            written.
        :type write_path: path-like or str

        :returns: A tuple of initialised variables used for simulation.
        :type: tuple(:class:`numpy.ndarray`, :class:`numpy.ndarray`,
                     :class:`numpy.ndarray`, :class:`numpy.ndarray`,
                     :class:`numpy.ndarray`, :class:`numpy.ndarray`,
                     :class:`numpy.ndarray`, :class:`numpy.ndarray`,
                     :class:`numpy.ndarray`, :class:`numpy.ndarray`,
                     :class:`numpy.ndarray`, :class:`numpy.ndarray`,
                     :class:`numpy.ndarray`, :class:`numpy.ndarray`,
                     :class`pathlib.Path`)
        """
        # Create initial displacements is none is provided
        if u is None:
            u = np.zeros((self.nnodes, 3))
        if ud is None:
            ud = np.zeros((self.nnodes, 3))
        damage = np.empty(self.nnodes).astype(np.float64)
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
        # Use the initial regimes of linear elastic (0 values) if none
        # is provided
        if regimes is None:
            regimes = np.zeros(
                (self.nnodes, self.max_neighbours), dtype=np.intc)
        elif type(regimes) == np.ndarray:
            if np.shape(regimes) != (self.nnodes, self.max_neighbours):
                raise ValueError("regimes must be a :class `numpy.ndarray`: \
                                 of shape (`nnodes`, `max_neighbours`)")
            regimes = regimes.astype(np.intc)
        else:
            raise TypeError("regimes must be a :class `numpy.ndarray`: or \
                            None")
        # Write down the initial connectivity in a file
        # Generate material types from connectivity and write to file
        # Generate stiffness correction factors and write to file

        # also define the 'tip' (for plotting displacements)
        # Initiate boundary condition containers
        bc_types = np.zeros(
            (self.nnodes, self.degrees_freedom), dtype=np.intc)
        bc_values = np.zeros(
            (self.nnodes, self.degrees_freedom), dtype=np.float64)
        force_bc_types = np.zeros(
            (self.nnodes, self.degrees_freedom), dtype=np.intc)
        force_bc_values = np.zeros(
            (self.nnodes, self.degrees_freedom), dtype=np.float64)
        tip_types = np.zeros(self.nnodes, dtype=np.intc)

        # Find the boundary nodes and apply the displacement values
        # Find the force boundary nodes and find amount of boundary nodes
        # TODO: generalise displacement boundary to 3D
        num_force_bc_nodes = 0
        for i in range(self.nnodes):
            # Define boundary types and values
            bnd = is_boundary(self.horizon, self.coords[i][:])
            # Define forces boundary types and values
            forces_bnd = is_forces_boundary(self.horizon, self.coords[i][:])
            if -1 in forces_bnd:
                num_force_bc_nodes += 1
            elif 1 in forces_bnd:
                num_force_bc_nodes += 1
            for j in range(self.degrees_freedom):
                forces_bnd_j = forces_bnd[j]
                bc_types[i, j] = np.intc((bnd))
                bc_values[i, j] = np.float64(bnd * displacement_rate)
                force_bc_types[i, j] = np.intc(forces_bnd_j)
                if forces_bnd_j != 2:
                    force_bc_values[i, j] = forces_bnd_j * max_reaction / (
                        self.volume[i])
            # Define tip #TODO: generalise to 3D
            tip = is_tip(self.horizon, self.coords[i][:])
            tip_types[i] = np.intc(tip)
        if num_force_bc_nodes != 0:
            force_bc_values = np.float64(
                np.divide(force_bc_values, num_force_bc_nodes))
        plus_cs = self._set_plus_cs(
            bond_stiffness, critical_stretch, self.n_regimes, self.n_materials)

        # If no write path was provided use the current directory, otherwise
        # ensure write_path is a Path object.
        if write_path is None:
            write_path = pathlib.Path()
        else:
            write_path = pathlib.Path(write_path)

        return (nlist, n_neigh, regimes, bond_stiffness, critical_stretch,
                plus_cs, u, ud, damage, bc_types, bc_values, force_bc_types,
                force_bc_values, tip_types, write_path)


class ContextError(Exception):
    """No suitable context was found by :func:`get_context`."""

    def __init__(self):
        """Exception constructor."""
        message = ("No suitable context was found. You can manually specify"
                   "the context by passing it to ModelCL with the 'context'"
                   "argument.")
        super().__init__(message)


def _calc_midpoint_gradient(T, displacement_scale_rate):
    """
    Calculate the midpoint gradient and coefficients of a 5th order polynomial.

    Calculates the midpoint gradient and coefficients of a 5th order
    polynomial displacement-time curve which is defined by acceleration being
    0 at t=0 and t=T and a midpoint gradient.

    :arg int T: The finish time-step of the displacement-time curve.
    :arg float displacement_scale_rate: The midpoint gradient of the curve.

    :returns: A tuple containing a float
        the midpoint gradient of the displacement-time curve and a tuple
        containing the 3 unconstrained coefficients of the 5th-order
        polynomial.
    :rtype: A tuple containing (:type float:, :type tuple:)
    """
    A = np.array([
        [(1 * T**5) / 1, (1 * T**4) / 1, (1 * T**3) / 1],
        [(20 * T**3) / 1, (12 * T**2) / 1, (6 * T**1) / 1],
        [(5 * T**4) / 1, (4 * T**3) / 1, (3 * T**2) / 1]
        ]
        )
    b = np.array(
        [
            [displacement_scale_rate],
            [0.0],
            [0.0]
                ])
    x = np.linalg.solve(A, b)
    a = x[0][0]
    b = x[1][0]
    c = x[2][0]
    midpoint_gradient = (5./16)*a*T**4 + (4./8)*b*T**3 + (3./4)*c*T**2
    coefficients = (a, b, c)
    return(midpoint_gradient, coefficients)


def _calc_build_time(build_displacement, displacement_rate, steps):
    """
    Calculate the the number of steps for the 5th order polynomial.

    An iterative procedure to calculate the number of steps over which the
    displacement-time curve is a smooth 5th order polynomial.

    :arg float build_displacement: The displacement in [m] over which the
        displacement-time graph is the smooth 5th order polynomial.
    :arg float displacement_rate: The displacement rate in [m] per step
            during the linear phase of the displacement-time graph.
    :arg int step: The current time-step of the simulation.

    :returns: A tuple containing an int T the number of steps over which the
        displacement-time curve is a smooth 5th order polynomial and a tuple
        containing the 3 unconstrained coefficients of the 5th-order
        polynomial.
    :rtype: A tuple containing (:type int:, :type tuple:)
    """
    build_time = 0
    midpoint_gradient = np.inf
    while midpoint_gradient > displacement_rate:
        # Try to calculate gradient, if not, increase the build_time
        try:
            midpoint_gradient, coefficients = _calc_midpoint_gradient(
                build_time, build_displacement)
        except Exception:
            pass
        build_time += 1
        if build_time > steps:
            # TODO: suggest some valid values from the parameters given
            raise ValueError(
                'Displacement build-up time was larger than total simulation \
                time steps! \ntry decreasing build_displacement, or increase \
                    max_displacement_rate. steps = {}'.format(steps))
            break
    return(build_time, coefficients)


def _calc_displacement_scale(
        coefficients, final_displacement, build_time, displacement_rate, step,
        build_displacement, ease_off):
    """
    Calculate the displacement scale.

    Calculates the displacement boundary condition scale according to a
    5th order polynomial/ linear displacement-time curve for which initial
    acceleration is 0.

    :arg tuple coefficients: Tuple containing the 3 free coefficients
        of the 5th order polynomial.
    :arg float final_displacement: The final applied displacement in [m].
    :arg int build_time: The number of time steps over which the
        applied displacement-time curve is not linear.
    :arg float displacement_rate: The displacement rate in [m] per step
        during the linear phase of the displacement-time graph.
    :arg int step: The current time-step of the simulation.
    :arg float build_displacement: The displacement in [m] over which the
        displacement-time graph is the smooth 5th order polynomial.
    :arg int ease_off: A boolean-like variable which is 0 if the
        displacement-rate hasn't started decreasing yet. Equal to the step
        at which the displacement rate starts decreasing once it does so.

    :returns: The displacement_scale between [0.0, 1.0], a scale
        applied to the displacement boundary conditions.
    :rtype: np.float64
    """
    a, b, c = coefficients
    if step < build_time / 2:
        m = 5 * a * step**4 + 4 * b * step**3 + 3 * c * step**2
        displacement_scale = m / displacement_rate
    elif ease_off != 0:
        t = step - ease_off + build_time / 2
        if t > build_time:
            displacement_scale = 0.0
        else:
            m = 5 * a * t**4 + 4 * b * t**3 + 3 * c * t**2
            displacement_scale = m / displacement_rate
    else:  # linear increments
        # calculate displacement
        linear_time = step - build_time/2
        linear_displacement = linear_time * displacement_rate
        displacement = linear_displacement + build_displacement/2
        if displacement + build_displacement / 2 < final_displacement:
            displacement_scale = 1.0
        else:
            ease_off = step
            displacement_scale = 1.0
    return(displacement_scale, ease_off)


# TODO: move to a utility package?
def output_device_info(device_id):
    """Output the device info of the device."""
    sys.stdout.write("Device is ")
    sys.stdout.write(device_id.name)
    if device_id.type == cl.device_type.GPU:
        sys.stdout.write("GPU from ")
    elif device_id.type == cl.device_type.CPU:
        sys.stdout.write("CPU from ")
    else:
        sys.stdout.write("non CPU of GPU processor from ")
    sys.stdout.write(device_id.vendor)
    sys.stdout.write(" with a max of ")
    sys.stdout.write(str(device_id.max_compute_units))
    sys.stdout.write(" compute units, \n")
    sys.stdout.write("a max of ")
    sys.stdout.write(str(device_id.max_work_group_size))
    sys.stdout.write(" work-items per work-group, \n")
    sys.stdout.write("a max work item dimensions of ")
    sys.stdout.write(str(device_id.max_work_item_dimensions))
    sys.stdout.write(", \na max work item sizes of ")
    sys.stdout.write(str(device_id.max_work_item_sizes))
    sys.stdout.write(",\nand device local memory size is ")
    sys.stdout.write(str(device_id.local_mem_size))
    sys.stdout.write(" bytes. \n")
    sys.stdout.flush()

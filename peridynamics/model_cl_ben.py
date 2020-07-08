"""Peridynamics model using Ben's Optimised OpenCL kernels."""
from .model import Model
from .cl_ben import double_fp_support, get_context, kernel_source
import numpy as np
import pyopencl as cl
from pyopencl import mem_flags as mf
import pathlib
from tqdm import trange
import sys # for output device info 

class ModelCLBen(Model):
    """A peridynamics model using Ben's optimised OpenCL code.

    This class allows users to define an composite peridynamics system with
    any number of materials and damage laws from parameters and a set of
    initial conditions (coordinates, connectivity, material_types and
    stiffness_correction_factors)."""
    
    def __init__(self,
                 *args,
                 density = None,
                 bond_stiffness = None,
                 critical_stretch = None,
                 bond_type = None,
                 material_types=None, 
                 stiffness_correction_factors=None,
                 precise_stiffness_correction = None,
                 dt = None,
                 write_path = None,
                 context=None, **kwargs):
        """Create a :class:`ModelCLBen` object.
        :arg float density: Density of the bulk material in kg/m^3.
        :arg float bond_stiffness: An (m, n_regimes) array of the different 
            bond stiffnesses for m material types.
        :arg float critical_stretch: An (m, n_regimes) array of 
            different critical strains for m material types.
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
        :arg stiffness_correction_factors: The stiffness_correction_factors for
            the model. If `None` the stiffness_correction_factors at the time
            of construction of the :class:`Model` object will be used. Default 
            `None`.
        :type stiffness_correction_factors:
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
        :arg write_path: The path where the stiffness_correction_factors,
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
        if len(bond_stiffness) != len(critical_stretch):
            raise ValueError("number of bond stiffnesses must be equal to\
                             the number of critical stretches")
        else:
            self.n_regimes = len(bond_stiffness)
        self.critical_stretch = critical_stretch
        self.bond_stiffness = bond_stiffness
        self.dt = dt

        if stiffness_correction_factors is None:
            # Calculate stiffness correction factors and write to file
            self.stiffness_correction_factors = self._set_stiffness_correction_factors(
                self.horizon, self.initial_connectivity, precise_stiffness_correction, 
                write_path)
        elif type(stiffness_correction_factors) == np.ndarray:
            if np.shape(stiffness_correction_factors) != (self.nnodes, self.max_neighbours):
                raise ValueError("stiffness_correction_factors must be of shape\
                                 (nnodes, max_neighbours)")
            else:
                self.stiffness_correction_factors = stiffness_correction_factors
        else:
            raise TypeError("stiffness_correction_factors must be a numpy.ndarray or None")

        # Create dummy boundary conditions function is none is provided
        if bond_type is None:
            def bond_type(x, y):
                return 0

        if material_types is None:
            # Calculate material types and write to file
            self.material_types = self._set_material_types(self.initial_connectivity, 
                                                      bond_type, write_path)
        elif type(material_types) == np.ndarray:
            if np.shape(material_types) != (self.nnodes, self.max_neighbours):
                raise ValueError("material_types must be of shape\
                                 (nnodes, max_neighbours)")
            else:
                self.material_types = material_types
        else:
            raise TypeError("stiffness_correction_factors must be a numpy.ndarray or None")

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
            + "-DPD_DPN_NODE_NO=" + str(self.degrees_freedom * self.nnodes) + SEP
            + "-DPD_NODE_NO=" + str(self.nnodes) + SEP
            + "-DMAX_HORIZON_LENGTH=" + str(self.max_neighbours) + SEP
            + "-DPD_DT=" + str(self.dt) + SEP
            + "-DPD_REGIME_NO=" + str(self.n_regimes) + SEP)

        # Build kernels
        self.program = cl.Program(self.context, kernel_source).build([options_string])
        self.queue = cl.CommandQueue(self.context)

        self.bond_force_kernel = self.program.bond_force_new
        self.update_displacement_kernel = self.program.update_displacement
        self.damage_kernel = self.program.damage
        self.damage_new_kernel = self.program.damage_new

    def _damage(self, nlist_d, family_d, damage_d, local_mem):
        """Calculate bond damage."""
        queue = self.queue

        # Call kernel
        self.damage_kernel(self.queue, (self.nnodes * self.max_horizon_length,),
                                  (self.max_horizon_length,), nlist_d,
                                           family_d, damage_d, local_mem)
        queue.finish()

    def _damage_new(self, n_neigh_d, family_d, damage_d):
        """Calculate bond damage."""
        queue = self.queue

        # Call kernel
        self.damage_kernel(queue, (self.nnodes,), None, n_neigh_d, family_d,
                           damage_d)
        queue.finish()

    def _bond_force(self, u_d, ud_d, r0_d, vols_d, nlist_d,
                    n_neigh_d, stiffness_correction_factors_d,
                    material_types_d, regimes_d, bond_stiffness_d, critical_stretch_d,
                    plus_cs_d,
                    force_bc_types_d, force_bc_values_d, local_mem_x, local_mem_y, local_mem_z, 
                    force_load_scale):
        """Calculate the force due to bonds acting on each node."""
        queue = self.queue
        # Call kernel
        self.bond_force_kernel(
                queue, (self.nnodes * self.max_horizon_length,), (self.max_horizon_length,), 
                u_d,
                ud_d,
                r0_d,
                vols_d,
                nlist_d,
                n_neigh_d,
                stiffness_correction_factors_d,
                material_types_d,
                regimes_d,
                bond_stiffness_d,
                critical_stretch_d,
                plus_cs_d,
                force_bc_types_d,
                force_bc_values_d,
                local_mem_x,
                local_mem_y,
                local_mem_z,
                force_load_scale
                )
        queue.finish()
        return ud_d, nlist_d, n_neigh_d, regimes_d

    def _update_displacement(self, ud_d, u_d, bc_types_d, bc_values_d, displacement_load_scale):
        """Update displacements."""
        # Call kernel
        self.update_displacement_kernel(
                self.queue, (self.degrees_freedom * self.nnodes,), None,
                ud_d,
                u_d,
                bc_types_d,
                bc_values_d,
                displacement_load_scale
                )
        return u_d

    def write_array(write_path, array):
        """
        :arg write_path: TYPE
        :type write_path: string or PATHFILE
        :arg numpy.ndarray array:
        """
        f = open(write_path, "w")
        f.write("# vtk DataFile Version 2.0\n")
        f.write("ASCII\n")
        f.write("\n")
        f.write("DATASET ARRAY\n")
        
        if type(array[0][0]) is (float or np.float64):
            for i in range(0, np.shape(array)[0]):
                tmp = array[i]
                for j in range(0, len(tmp)):
                    f.write("{:f} ".format(tmp[j]))
                f.write("\n")
        elif type(array[0][0]) is (int or np.intc):
            for i in range(0, np.shape(array)[0]):
                tmp = array[i]
                for j in range(0, len(tmp)):
                    f.write("{:d} ".format(np.intc(tmp[j])))
                f.write("\n")
        else:
            ValueError('values','type not recognised, could not write to file.\
                       Type must be float, numpy.float64, int or numpy.intc')

    def _set_material_types(self, initial_connectivity, bond_type, write_path):
        """
        Builds a list of indices for the material types. The indices are used
        to index into the bond_stiffness and critical_stretch.
        
        :arg initial_connectivity:
        :arg bond_type:
        :arg write_path:
        :return:
        :rtype:
        """
        nlist, n_neigh = initial_connectivity
        material_types = np.zeros((self.nnodes, self.max_neighbours), dtype=np.intc)
        for i in range(self.nnodes):
            for neighbour in range(n_neigh[i]):
                j = nlist[i][neighbour]
                material_types[i][j] = bond_type(self.coords[i, :], self.coords[j, :])
        material_types = material_types.astype(np.intc)
        self.write_array(write_path/"material_types", material_types)
        return material_types

    def _set_stiffness_correction_factor(self, horizon, initial_connectivity,
                                         precise_stiffness_correction, write_path):
        """
        Builds a list of stiffness correction factors that reduce the peridynamic
        surface softening effect for 2D/3D problem and writes to file. The 'volume
        method' proposed in Chapter 2 in Bobaru F, Foster JT, Geubelle PH, 
        Silling SA (2017) Handbook of peridynamic modeling (p 51–52) is used here.

        :arg float horizon: The horizon distance.
        :arg int size: The size of each row of the neighbour list. This is the
            maximum number of neighbours and should be equal to the maximum of
            of :func:`peridynamics.neighbour_list.family`.
        :arg precise_stiffness_correction int: A switch variable. (=1), Calculate precise
                                                stiffening factor more accurately using 
                                                actual nodal volumes.
                                                (=0),Calculate non-precise stiffening
                                                factor using average nodal volumes.
                                                (=None), Don't apply a stiffness 
                                                correction factor.
        :return: A tuple of the neighbour list and number of neighbours for each
            node.
        :rtype: numpy.ndarray
        """
        
        nlist, n_neigh = initial_connectivity
        stiffness_correction_factors = np.ones((self.nnodes, n_neigh))
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

        if precise_stiffness_correction ==1:
            for i in range(0, self.nnodes):
                neighbour_list = nlist[i][:self.family[i]]
                family_volume_i = family_volumes[i]
                for j in range(self.family[i]):
                    family_volume_j = family_volumes[neighbour_list[j]]
                    stiffness_correction_factor = 2.* family_volume_bulk / \
                    (family_volume_i + family_volume_j)
                    stiffness_correction_factors[i][j]= stiffness_correction_factor

        elif precise_stiffness_correction == 0:
            average_node_volume = self.volume_total/self.nnodes
            for i in range(0, self.nnodes):
                nnodes_i_family = self.family[i]
                nodei_family_volume = nnodes_i_family * average_node_volume # Possible to calculate more exactly, we have the volumes for free
                for j in nnodes_i_family:
                    nnodes_j_family = self.family[j]
                    nodej_family_volume = nnodes_j_family* average_node_volume # Possible to calculate more exactly, we have the volumes for free
                    stiffness_correction_factor = 2.* family_volume_bulk /  (nodej_family_volume + nodei_family_volume)
                    stiffness_correction_factors[i][j]= stiffness_correction_factor

        elif precise_stiffness_correction == None:
            pass
        else:
            raise ValueError('precise_stiffness_correction', 
                             'precise_stiffness_correction can \
                             only take values 0 or 1 or None')
        self.write_array(write_path/"stiffness_correction_factors", stiffness_correction_factors)
        return stiffness_correction_factors

    def _set_plus_cs(self, bond_stiffness, critical_stretch, n_regimes):
        """

        Parameters
        ----------
        bond_stiffness : TYPE
            DESCRIPTION.
        critical_stretch : TYPE
            DESCRIPTION.
        n_regimes : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        # For initial elastic regime, the bond force density at 0 stretch is 0
        c0 = 0.0
        c_prev = c0
        plus_cs = [c0]
        if n_regimes != 1:
            
            for i in range(n_regimes -1):
                c_i = c_prev + bond_stiffness[i-1]*critical_stretch[i-1] -\
                    bond_stiffness[i]*critical_stretch[i-1]
                plus_cs.append[c_i]
                c_prev = c_i
        assert len(plus_cs) == n_regimes
        return plus_cs
        
    def _increment_load(self, load_scale):
        """
        

        Parameters
        ----------
        load_scale : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        if self.num_force_bc_nodes != 0:
            # update the host force load scale
            self.force_load_scale_h = np.float64(load_scale)
    def _increment_displacement(self, coefficients, build_time, step, ease_off,
                                displacement_rate, build_displacement ,
                                final_displacement):
        """
        Increments the displacement boundary condition values depending on the
        time step, according to the function parameters, displacement_rate, 
        build_displacement and final_displacement.
        :arg np.ndarray coefficients: discription
        :arg int? build_time:
        :arg int step:
        :arg bool ease_off:
        :arg float displacement_rate:
        :arg float build_displacement:
        :arg float final_displacement:

        :return: displacement_load_scale
        :rtype:
        :return ease_off
        :rtype:
        """
        if not ((displacement_rate is None) or (build_displacement is None)\
                or (final_displacement is None)):
            # 5th order polynomial/ linear curve used to calculate displacement_scale
            displacement_scale, ease_off = _calc_load_displacement_rate(coefficients,
                                                             final_displacement,
                                                             build_time,
                                                             displacement_rate,
                                                             step, 
                                                             build_displacement,
                                                             ease_off)
            if displacement_scale != 0.0:
                # update the host force load scale
                displacement_load_scale = np.float64(displacement_scale)
        # No user specified build up parameters case
        elif not (displacement_rate is None):
            # update the host force load scale
            displacement_load_scale = np.float64(1.0)
        return displacement_load_scale, ease_off

    def simulate(self, steps, u=None,
                 ud=None, connectivity=None,
                 bond_stiffness= None,
                 critical_stretch=None,
                 is_forces_boundary=None,
                 is_boundary=None,      
                 is_tip=None,
                 displacement_rate=None, build_displacement=None,
                 final_displacement=None, 
                 build_load = None,
                 max_reaction = None,
                 first_step=1, write=None, write_path=None):
        """
        Simulate the peridynamics model.
        :arg int steps: The number of simulation steps to conduct.
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
        :arg ud: The initial velocities for the simulation. If `None` the
            velocities will be initialised to zero. Default `None`.
        :type ud: :class:`numpy.ndarray`
        :arg ud: The initial velocities for the simulation. If `None` the
            velocities will be initialised to zero. Default `None`.
        :type ud: :class:`numpy.ndarray`
        :arg displacement_rate:
        :type displacement_rate:
        :arg build_displacement:
        :type build_displacement:
        :arg final_displacement:
        :type final_displacement:
        :arg build_load:
        :type build_load:
        :arg max_reaction:
        :type max_reaction:
        :arg int first_step: The starting step number. This is useful when
            restarting a simulation, especially if `boundary_function` depends
            on the absolute step number.
        :arg int write: The frequency, in number of steps, to write the system
            to a mesh file by calling :meth:`Model.write_mesh`. If `None` then
            no output is written. Default `None`.
        :arg write_path: The path where the periodic mesh files should be
            written.
        :type write_path: path-like or str
        """
        (nlist, n_neigh, regimes, bond_stiffness, critical_stretch,
         plus_cs,
         u,
         ud,
         damage,
         bc_types,
         bc_values,
         force_bc_types,
         force_bc_values,
         tip_types,
         write_path) = self._simulate_initialise(is_forces_boundary, is_boundary, is_tip, displacement_rate,
                             u, ud, connectivity, 
                             bond_stiffness, critical_stretch, write_path)
             
        # Calculate number of time steps that displacement load is in the 'build-up' phase
        if not ((displacement_rate is None) or (build_displacement is None) or (final_displacement is None)):
            build_time, coefficients = _calc_build_time(build_displacement, displacement_rate, steps)

        # Local memory containers for Bond forces
        local_mem_x = cl.LocalMemory(np.dtype(np.float64).itemsize * self.max_neighbours)
        local_mem_y = cl.LocalMemory(np.dtype(np.float64).itemsize * self.max_neighbours)
        local_mem_z = cl.LocalMemory(np.dtype(np.float64).itemsize * self.max_neighbours)

        # For applying force in incriments
        force_load_scale = np.float64(0.0)
        # For applying displacement in incriments
        displacement_load_scale = np.float64(0.0)

        # Build OpenCL data structures

        # Read only
        r0_d = cl.Buffer(self.context,
                             cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                             hostbuf=np.ascontiguousarray(self.coords, dtype=np.float64))
        bc_types_d = cl.Buffer(self.context,
                              cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                              hostbuf=bc_types)
        bc_values_d = cl.Buffer(self.context,
                               cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                               hostbuf=bc_values)
        force_bc_types_d = cl.Buffer(self.context,
                              cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                              hostbuf=force_bc_types)
        force_bc_values_d = cl.Buffer(self.context,
                               cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                               hostbuf=force_bc_values)
        vols_d = cl.Buffer(self.context,
                           cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                           hostbuf=self.volume)
        stiffness_correction_factors_d = cl.Buffer(self.context,
                           cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                           hostbuf=np.ascontiguousarray(self.stiffness_correction_factors, dtype=np.float64))
        material_types_d = cl.Buffer(self.context,
                           cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                           hostbuf=np.ascontiguousarray(self.material_types, dtype=np.float64))
        bond_stiffness_d = cl.Buffer(self.context,
                           cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                           hostbuf=np.ascontiguousarray(bond_stiffness, dtype=np.float64))
        critical_stretch_d = cl.Buffer(self.context,
                           cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                           hostbuf=np.ascontiguousarray(critical_stretch, dtype=np.float64))
        plus_cs_d = cl.Buffer(self.context,
                           cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                           hostbuf=np.ascontiguousarray(plus_cs, dtype=np.float64))
        family_d = cl.Buffer(
                self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                hostbuf=self.family)
        # Read and write
        regimes_d = cl.Buffer(self.context,
                           cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR,
                           hostbuf=np.ascontiguousarray(regimes, dtype=np.float64))
        nlist_d = cl.Buffer(
                self.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR,
                hostbuf=nlist)
        n_neigh_d = cl.Buffer(
                self.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR,
                hostbuf=n_neigh)
        u_d = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, np.empty((self.nnodes, self.degrees_freedom), dtype=np.float64).nbytes)
        ud_d = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, np.empty((self.nnodes, self.degrees_freedom), dtype=np.float64).nbytes)

        # Write only
        damage_d = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY, np.empty(self.nnodes).astype(np.float64).nbytes)
        # Initialize kernel parameters TODO: no sign of this in Jim's code, is it relevant?
        self.bond_force_kernel.set_scalar_arg_dtypes(
            [None,
             None,
             None,
             None,
             None,
             None,
             None,
             None,
             None,
             None,
             None,
             None,
             None,
             None,
             None,
             None,
             None,
             None
             ])
        # Initialize kernel parameters
        self.update_displacement_kernel.set_scalar_arg_dtypes(
            [None,
             None,
             None,
             None,
             None
             ])
        self.damage_kernel.set_scalar_arg_dtypes(
            [None, None, None, None])
        self.damage_kernel.set_scalar_arg_dtypes(
            [None, None, None])

        # Container for plotting data
        damage_sum_data = []
        tip_displacement_data = []
        tip_force_data = []

        # Ease off displacement loading switch
        ease_off = 0
        for step in trange(first_step, first_step+steps,
                           desc="Simulation Progress", unit="steps"):
            
            # Update displacements
            u_d = self._update_displacement(ud_d, u_d, bc_types_d, bc_values_d, displacement_load_scale)
                
            # Calculate the force due to bonds on each node, and update connectivity
            ud_d, nlist_d, n_neigh_d, regimes_d = self._bond_force(u_d, ud_d, r0_d, vols_d, nlist_d, 
                                             n_neigh_d, stiffness_correction_factors_d,
                                             material_types_d, regimes_d, bond_stiffness_d,
                                             critical_stretch_d, plus_cs_d,
                                             force_bc_types_d, 
                                             force_bc_values_d, local_mem_x, local_mem_y,
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
                        if self.tip_types[i] == 1:
                            tmp +=1
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
                        print('Warning: over 2% of bonds have broken! -- PERIDYNAMICS SIMULATION CONTINUING')
                    elif damage_sum > 0.7*self.nnodes:
                        print('Warning: over 7% of bonds have broken! -- PERIDYNAMICS SIMULATION STOPPING')
                        break

            # Increase load in linear increments
            if not (build_load is None):
                load_scale = min(1.0, self.build_load * step)
                if load_scale != 1.0:
                    self._increment_load(load_scale)
            # Increase dispalcement in 5th order polynomial increments
            displacement_load_scale = self._increment_displacement(
                coefficients, build_time, step, ease_off, displacement_rate, 
                build_displacement, final_displacement)

        return damage_sum_data, tip_displacement_data, tip_force_data

    def _simulate_initialise(self, is_forces_boundary, is_boundary, is_tip, displacement_rate,
                             max_reaction,
                             u, ud, connectivity, 
                             bond_stiffness, critical_stretch, write_path):
        """
        Initialise simulation variables.

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
        :arg write_path: The path where the periodic mesh files should be
            written.
        :type write_path: path-like or str

        :returns: A tuple of initialised variables used for simulation.
        :type: tuple(:class:`numpy.ndarray`, :class:`numpy.ndarray`,
            :class:`numpy.ndarray`, function, :class`pathlib.Path`)
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
        # Write down the initial connectivity in a file
        # Generate material types from connectivity and write to file
        # Generate stiffness correction factors and write to file
        
        #also define the 'tip' (for plotting displacements)
        # Initiate boundary condition containers
        bc_types = np.zeros((self.nnodes, self.degrees_freedom), dtype=np.intc)
        bc_values = np.zeros((self.nnodes, self.degrees_freedom), dtype=np.float64)
        force_bc_types = np.zeros((self.nnodes, self.degrees_freedom), dtype=np.intc)
        force_bc_values = np.zeros((self.nnodes, self.degrees_freedom), dtype=np.float64)
        tip_types = np.zeros(self.nnodes, dtype=np.intc)

        # Find the boundary nodes and apply the displacement values
        # Find the force boundary nodes and find amount of boundary nodes
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
                bc_values[i, j] = np.float64(bnd[j] * displacement_rate)
                force_bc_types[i, j] = np.intc(forces_bnd_j)
                if forces_bnd_j != 2:
                    force_bc_values[i, j] = forces_bnd_j * max_reaction / (self.volume[i])
            # Define tip #TODO: generalise to 3D
            tip = is_tip(self.horizon, self.coords[i][:])
            tip_types[i] = np.intc(tip)
        force_bc_values = np.float64(np.divide(force_bc_values,num_force_bc_nodes))
        regimes = np.zeros((self.nnodes, self.max_neighbours), dtype=np.intc)
        plus_cs = self._set_plus_cs(bond_stiffness, critical_stretch, self.n_regimes)

        # If no write path was provided use the current directory, otherwise
        # ensure write_path is a Path object.
        if write_path is None:
            write_path = pathlib.Path()
        else:
            write_path = pathlib.Path(write_path)

        return (nlist, n_neigh, regimes, bond_stiffness, critical_stretch,
         plus_cs,
         u,
         ud,
         damage,
         bc_types,
         bc_values,
         force_bc_types,
         force_bc_values,
         tip_types,
         write_path)

class ContextError(Exception):
    """No suitable context was found by :func:`get_context`."""

    def __init__(self):
        """Exception constructor."""
        message = ("No suitable context was found. You can manually specify"
                   "the context by passing it to ModelCL with the 'context'"
                   "argument.")
        super().__init__(message)

def _calc_midpoint_gradient(T, displacement_scale_rate):
    A = np.array([
        [(1*T**5)/1,(1*T**4)/1,(1*T**3)/1],
        [(20*T**3)/1,(12*T**2)/1,(6*T**1)/1],
        [(5*T**4)/1,(4*T**3)/1,(3*T**2)/1,]
        ]
        )
    b = np.array(
        [
            [displacement_scale_rate],
            [0.0],
            [0.0]
                ])
    x = np.linalg.solve(A,b)
    a = x[0][0]

    b = x[1][0]

    c = x[2][0]
    
    midpoint_gradient = (5./16)*a*T**4 + (4./8)*b*T**3 + (3./4)*c*T**2
    coefficients = (a, b, c)
    return(midpoint_gradient, coefficients)

def _calc_build_time(build_displacement, displacement_scale_rate, steps):
    T = 0
    midpoint_gradient = np.inf
    while midpoint_gradient > displacement_scale_rate:
        try:
            midpoint_gradient, coefficients = _calc_midpoint_gradient(
                T, build_displacement)
        except:
            pass
        T += 1
        if T > steps:
            # TODO: suggest some valid values from the parameters given
            raise ValueError(
                'Displacement build-up time was larger than total simulation \
                time steps! \ntry decreasing build_displacement, or increase \
                    max_displacement_rate. steps = {}'.format(steps))
            break
    return(T, coefficients)

def _calc_load_displacement_rate(coefficients, final_displacement, build_time,
                                 displacement_scale_rate, step, 
                                 build_displacement, ease_off):
    a, b, c = coefficients
    if step < build_time/2:
        m = 5*a*step**4 + 4*b*step**3 + 3*c*step**2
        #print('m = ', m)
        load_displacement_rate = m/displacement_scale_rate
    elif ease_off != 0:
        t = step - ease_off + build_time/2
        if t > build_time:
            load_displacement_rate = 0.0
        else:
            m = 5*a*t**4 + 4*b*t**3 + 3*c*t**2
            load_displacement_rate = m/displacement_scale_rate
    else: # linear increments
        # calculate displacement
        linear_time = step - build_time/2
        linear_displacement = linear_time * displacement_scale_rate
        displacement = linear_displacement + build_displacement/2
        if displacement + build_displacement/2 < final_displacement:
            load_displacement_rate = 1.0
        else:
            ease_off = step
            load_displacement_rate = 1.0
    return(load_displacement_rate, ease_off)

def output_device_info(device_id):
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
        
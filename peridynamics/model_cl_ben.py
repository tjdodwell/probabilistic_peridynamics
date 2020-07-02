"""Peridynamics model using Ben's Optimised OpenCL kernels."""
from .model import Model
from .cl_ben import double_fp_support, get_context, kernel_source
import numpy as np
import pyopencl as cl
from pyopencl import mem_flags as mf
from peridynamics.post_processing import vtk # For _set_network
import pathlib
import sys # for output device info 

class ModelCLBen(Model):
    """A peridynamics model using Ben's optimised OpenCL code.

    This class allows users to define a 2 material peridynamics system from parameters
    and a set of initial conditions (coordinates and connectivity)."""
    
    def __init__(self,
                 *args,
                 density = None,
                 bond_stiffness_concrete = None,
                 bond_stiffness_steel = None,
                 critical_strain_concrete = None,
                 critical_strain_steel = None,
                 bond_type = None,
                 volume_total=None,
                 network_file_name = 'Network.vtk',
                 transfinite = None,
                 precise_stiffness_correction = None,
                 max_reaction = None,
                 build_load = None,
                 displacement_rate = None,
                 damping = None,
                 build_displacement = None,
                 max_displacement = None,
                 initial_crack = [], # remove this
                 crack_length = None,
                 context=None, **kwargs):
        """Create a :class:`ModelCLBen` object.
        :arg float density: Density of the bulk material in kg/m^3 .
        :arg float bond_stiffness_concrete: The bond stiffness of concrete. (TMP)
        :arg float bond_stiffness_steel: The bond stiffness of steel. (TMP)
        :arg float critical_strain_concrete: The critical strain of concrete. (TMP)
        :arg float critical_strain_steel: The critical strain of steel. (TMP)
        :arg method bond_type: A method which outputs the material of the bond.
        :arg string network_file_name: Name of the network file defining the systems 
            horizons, horizon lengths and stiffness correction factors.
        :arg bool transfinite: Cartesian cubic (tensor grid) mesh (1) or 
            tetra-hedral grid (default, 0).
        :arg bool precise_stiffness_correction: Boolean for stiffness correction
            factors calculated using mesh element volumes (default 'precise', 1) or
            average nodal volume of a transfinite mesh (0).
        :arg float max_reaction: The maximum total load applied to the loaded nodes.
        :arg int build_load: The number of steps to apply the max reaction
            force to the loaded nodes at a linear rate.
        :arg int build_displacement: The number of steps to apply the build up for
            the displacement.
        :arg float max_displacement: The maximum displacement applied to the loaded
            nodes.
        :returns: A new :class:`Model` object.
        :rtype: Model
        """
        super().__init__(*args, **kwargs)

        self.degrees_freedom = 3
        self.precise_stiffness_correction = precise_stiffness_correction
        self.transfinite = transfinite
        self.density = density
        self.bond_stiffness_concrete = bond_stiffness_concrete
        self.bond_stiffness_steel = bond_stiffness_steel
        self.critical_strain_concrete = critical_strain_concrete
        self.critical_strain_steel = critical_strain_steel
        self.network_file_name = network_file_name
        self.dt = None
        self.max_reaction = None
        self.load_scale_rate = None

        self._set_volume(volume_total)

        # If the network has already been written to file, then read, if not, setNetwork
        try:
            self._read_network(self.network_file_name)
        except:
            print('No network file found: writing network file.')
            self._set_network(self.horizon, bond_type)

        # Initate crack
        self._set_connectivity(initial_crack)

        # Initiate boundary condition containers
        self.bc_types = np.zeros((self.nnodes, self.degrees_freedom), dtype=np.intc)
        self.bc_values = np.zeros((self.nnodes, self.degrees_freedom), dtype=np.float64)
        self.tip_types = np.zeros(self.nnodes, dtype=np.intc)

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
            + "-DMAX_HORIZON_LENGTH=" + str(self.max_horizon_length) + SEP
            + "-DPD_DT=" + str(self.dt) + SEP)

        # Build kernels
        self.program = cl.Program(self.context, kernel_source).build([options_string])
        self.queue = cl.CommandQueue(self.context)

        self.bond_force_kernel = self.program.bond_force
        self.update_displacement_kernel = self.program.update_displacement
        self.damage_kernel = self.program.damage

    def _damage(self, horizons_d, horizons_lengths_d, damage_d, local_mem):
        """Calculate bond damage."""
        queue = self.queue

        # Call kernel
        self.damage_kernel(self.queue, (self.nnodes * self.max_horizon_length,),
                                  (self.max_horizon_length,), self.horizons_d,
                                           self.horizons_lengths_d, damage_d, local_mem)
        queue.finish()

    def _bond_force(self, u_d, ud_d, r0_d, vols_d, horizons_d, bond_stiffness_d,
                    bond_critical_stretch_d, force_bc_types_d, force_bc_values_d,
                    local_mem_x, local_mem_y, local_mem_z, force_load_scale,
                    displacement_load_scale):
        """Calculate the force due to bonds acting on each node."""
        queue = self.queue
        # Call kernel
        self.bond_force_kernel(
                queue, (self.nnodes * self.max_horizon_length,), (self.max_horizon_length,), 
                u_d,
                ud_d,
                vols_d,
                horizons_d,
                r0_d,
                bond_stiffness_d,
                bond_critical_stretch_d,
                force_bc_types_d,
                force_bc_values_d,
                local_mem_x,
                local_mem_y,
                local_mem_z,
                force_load_scale,
                displacement_load_scale
                )
        queue.finish()

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

    def _set_connectivity(self, initial_crack):
        """
        Sets the intial crack.
        :arg initial_crack: The initial crack of the system. The argument may
            be a list of tuples where each tuple is a pair of integers
            representing nodes between which to create a crack. Alternatively,
            the arugment may be a function which takes the (nnodes, 3)
            :class:`numpy.ndarray` of coordinates as an argument, and returns a
            list of tuples defining the initial crack.
        :type initial_crack: list(tuple(int, int)) or function
        :returns: None
        :rtype: NoneType
        
        
        bb515 connectivity matrix is replaced by self.horizons and self.horizons_lengths for OpenCL
        
        also see self.family, which is a verlet list:
            self.horizons and self.horizons_lengths are neccessary OpenCL cannot deal with non fixed length arrays
        """
        if self.v == True:
            print("defining crack")
        # This code is the fastest because it doesn't have to iterate through
        # all possible initial crack bonds.
        def is_crack(x, y):
            output = 0
            crack_length = 0.3
            p1 = x
            p2 = y
            if x[0] > y[0]:
                p2 = x
                p1 = y
            # 1e-6 makes it fall one side of central line of particles
            if p1[0] < 0.5 + 1e-6 and p2[0] > 0.5 + 1e-6:
                # draw a straight line between them
                m = (p2[1] - p1[1]) / (p2[0] - p1[0])
                c = p1[1] - m * p1[0]
                # height a x = 0.5
                height = m * 0.5 + c
                if (height > 0.5 * (1 - crack_length)
                        and height < 0.5 * (1 + crack_length)):
                    output = 1
            return output

        for i in range(0, self.nnodes):
            for k in range(0, self.max_horizon_length):
                j = self.horizons[i][k]
                if is_crack(self.coords[i, :], self.coords[j, :]):
                    self.horizons[i][k] = np.intc(-1)

    def _read_network(self, network_file):
        """ For reading a network file if it has been written to file yet.
        Significantly quicker than building horizons from scratch, however
        the network file size is quite large for large node num.
        :arg network_file: the network vtk file including information about
        node families, bond stiffnesses, critical stretches, number of nodes
        , max horizon length and horizon lengths.
        """
        def find_string(string, iline):
            """
            Function for incrimenting the line when reading the vtk file,
            network_file until input string is found
            :
            :arg string: The string in the vtk to be found.
            :arg iline: The current count of the line no. in the read of
            'network_file'
            :returns: list of strings of row of the chosen line
            :rtype: list
            """
            found = 0
            while (found == 0):
                iline+= 1
                line = f.readline()
                row = line.strip()
                row_as_list = row.split()
                found = 1 if string in row_as_list else 0
            return row_as_list, iline

        f = open(network_file, "r")

        if f.mode == "r":
            iline = 0

            # Read the Max horizons length first
            row_as_list, iline = find_string('MAX_HORIZON_LENGTH', iline)
            max_horizon_length = int(row_as_list[1])
            if self.v == True:
                print('max_horizon_length', max_horizon_length)
            # Read nnodes
            row_as_list, iline = find_string('NNODES', iline)
            nnodes = int(row_as_list[1])
            if self.v == True:
                print('nnodes', nnodes)
            # Read horizons lengths
            row_as_list, iline = find_string('HORIZONS_LENGTHS', iline)
            horizons_lengths = np.zeros(nnodes, dtype=int)
            for i in range(0, nnodes):
                iline += 1
                line = f.readline()
                horizons_lengths[i] = np.intc(line.split())

            # Read family matrix
            if self.v == True:
                print('Building family matrix from file')
            row_as_list, iline = find_string('FAMILY', iline)
            family = []
            for i in range(nnodes):
                iline += 1
                line = f.readline()
                row = line.strip()
                row_as_list = row.split()
                family.append(np.zeros(len(row_as_list), dtype=np.intc))
                for j in range(0, len(row_as_list)):
                    family[i][j] = np.intc(row_as_list[j])

            # Read stiffness values
            if self.v == True:
                print('Building stiffnesses from file')
            row_as_list, iline = find_string('STIFFNESS', iline)
            bond_stiffness_family = []
            for i in range(nnodes):
                iline += 1
                line = f.readline()
                row = line.strip()
                row_as_list = row.split()
                bond_stiffness_family.append(np.zeros(len(row_as_list), dtype=np.float64))
                for j in range(0, len(row_as_list)):
                    bond_stiffness_family[i][j] = (row_as_list[j])

            # Now read critcal stretch values
            if self.v == True:
                print('Building critical stretch values from file')
            row_as_list, iline = find_string('STRETCH', iline)
            bond_critical_stretch_family = []
            for i in range(nnodes):
                iline += 1
                line = f.readline()
                row = line.strip()
                row_as_list = row.split()
                bond_critical_stretch_family.append(np.zeros(len(row_as_list), dtype=np.float64))
                for j in range(0, len(row_as_list)):
                    bond_critical_stretch_family[i][j] = row_as_list[j]

            # Maximum number of nodes that any one of the nodes is connected to
            max_horizon_length_check = np.intc(
                    1<<(len(max(family, key=lambda x: len(x)))-1).bit_length()
                )
            assert max_horizon_length == max_horizon_length_check, 'Read failed on MAX_HORIZON_LENGTH check'

            horizons = -1 * np.ones([nnodes, max_horizon_length])
            for i, j in enumerate(family):
                horizons[i][0:len(j)] = j

            bond_stiffness = -1. * np.ones([nnodes, max_horizon_length])
            for i, j in enumerate(bond_stiffness_family):
                bond_stiffness[i][0:len(j)] = j

            bond_critical_stretch = -1. * np.ones([nnodes, max_horizon_length])
            for i, j in enumerate(bond_critical_stretch_family):
                bond_critical_stretch[i][0:len(j)] = j

            # Make sure it is in a datatype that C can handle
            self.horizons = horizons.astype(np.intc)
            self.bond_stiffness = bond_stiffness
            self.bond_critical_stretch = bond_critical_stretch
            self.horizons_lengths = horizons_lengths.astype(np.intc)
            self.family = family
            self.max_horizon_length = max_horizon_length
            self.nnodes = nnodes
            f.close()

    def _set_volume(self, volume_total=None):
        """
        Calculate the value of each node.

        :arg volume_total: User input for the total volume of the mesh, for checking sum total of elemental volumes is equal to user input volume for simple prismatic problems.
        In the case of non-prismatic problems when the user does not know what the volume is, we should do something else as an assertion
        :returns: None
        :rtype: NoneType
        """
        # bb515 this has changed significantly from the sequential code.
        # OpenCL (or rather C) requires that we are careful with
        # types so that they are compatible with the specifed C types in the
        # OpenCL kernels
        self.V = np.zeros(self.nnodes, dtype=np.float64)

        # this is the sum total of the elemental volumes, initiated at 0.
        self.sum_total_volume = 0

        if self.transfinite == 1:
            """ Tranfinite mode is when we have approximated the volumes of the nodes
            as the average volume of nodes on a rectangular grid.
            The transfinite grid (search on youtube for "transfinite mesh gmsh") is not
            neccessarily made up of tetrahedra, but may be made up of cuboids.
            """
            tmp = volume_total / self.nnodes
            for i in range(0, self.nnodes):
                self.V[i] = tmp
                self.sum_total_volume += tmp
        else:
            for element in self.mesh_connectivity:

                # Compute Area or Volume
                val = 1. / len(element)

                # Define area of element
                if self.dimensions == 2:

                    xi, yi, *_ = self.coords[element[0]]
                    xj, yj, *_ = self.coords[element[1]]
                    xk, yk, *_ = self.coords[element[2]]

                    element_area = (
                            0.5 * np.absolute((
                                    (xj - xi) * (yk - yi) - (xk - xi) * (yj - yi)))
                            )
                    val *= element_area
                    self.sum_total_volume += element_area

                elif self.dimensions == 3:

                    a = self.coords[element[0]]
                    b = self.coords[element[1]]
                    c = self.coords[element[2]]
                    d = self.coords[element[3]]

                    # Volume of a tetrahedron
                    i = np.subtract(a,d)
                    j = np.subtract(b,d)
                    k = np.subtract(c,d)

                    element_volume = (1./6) * np.absolute(np.dot(i, np.cross(j,k)))
                    val*= element_volume
                    self.sum_total_volume += element_volume
                else:
                    raise ValueError('dim', 'dimension size can only take values 2 or 3')

                for j in range(0, len(element)):
                    self.V[element[j]] += val

        self.V = self.V.astype(np.float64)

    def _set_network(self, horizon, bond_type):
        """
        Sets the family matrix, and converts this to a horizons matrix 
        (a fixed size data structure compatible with OpenCL).
        Calculates horizons_lengths
        Also initiate crack here if there is one
        :arg horizon: Peridynamic horizon distance
        :returns: None
        :rtype: NoneType
        """
        def l2(y1, y2):
            """
            Euclidean distance between nodes y1 and y2.
            """
            l2 = 0
            for i in range(len(y1)):
                l2 += (y1[i] - y2[i]) * (y1[i] - y2[i])
            l2 = np.sqrt(l2)
            return l2

        # Container for nodal family
        family = []
        bond_stiffness_family = []
        bond_critical_stretch_family = []

        # Container for number of nodes (including self) that each of the nodes
        # is connected to
        self.horizons_lengths = np.zeros(self.nnodes, dtype=np.intc)

        for i in range(0, self.nnodes):
            print('node', i, 'networking...')
            # Container for family nodes
            tmp = []
            # Container for bond stiffnesses
            tmp2 = []
            # Container for bond critical stretches
            tmp3 = []
            for j in range(0, self.nnodes):
                if i != j:
                    distance = l2(self.coords[i, :], self.coords[j, :])
                    if distance < horizon:
                        tmp.append(j)
                        # Determine the material properties for that bond
                        material_flag = bond_type(self.coords[i, :], self.coords[j, :])
                        if material_flag == 'steel':
                            tmp2.append(self.bond_stiffness_steel)
                            tmp3.append(self.critical_strain_steel)
                        elif material_flag == 'interface':
                            tmp2.append(self.bond_stiffness_concrete * 3.0) # factor of 3 is used for interface bonds in the literature turn this off for parameter est. tests
                            tmp3.append(self.critical_strain_concrete * 3.0) # 3.0 is used for interface bonds in the literature
                        elif material_flag == 'concrete':
                            tmp2.append(self.bond_stiffness_concrete)
                            tmp3.append(self.critical_strain_concrete)

            family.append(np.zeros(len(tmp), dtype=np.intc))
            bond_stiffness_family.append(np.zeros(len(tmp2), dtype=np.float64))
            bond_critical_stretch_family.append(np.zeros(len(tmp3), dtype=np.float64))

            self.horizons_lengths[i] = np.intc((len(tmp)))
            for j in range(0, len(tmp)):
                family[i][j] = np.intc(tmp[j])
                bond_stiffness_family[i][j] = np.float64(tmp2[j])
                bond_critical_stretch_family[i][j] = np.float64(tmp3[j])

        assert len(family) == self.nnodes
        # As numpy array
        self.family = np.array(family)

        # Do the bond critical ste
        self.bond_critical_stretch_family = np.array(bond_critical_stretch_family)
        self.bond_stiffness_family = np.array(bond_stiffness_family)

        self.family_v = np.zeros(self.nnodes)
        for i in range(0, self.nnodes):
            tmp = 0 # tmp family volume
            family_list = family[i]
            for j in range(0, len(family_list)):
                tmp += self.V[family_list[j]]
            self.family_v[i] = tmp

        if self.precise_stiffness_correction == 1:
            # Calculate stiffening factor nore accurately using actual nodal volumes
            for i in range(0, self.nnodes):
                family_list = family[i]
                nodei_family_volume = self.family_v[i]
                for j in range(len(family_list)):
                    nodej_family_volume = self.family_v[j]
                    stiffening_factor = 2.* self.family_volume /  (nodej_family_volume + nodei_family_volume)
                    print('Stiffening factor {}'.format(stiffening_factor))
                    bond_stiffness_family[i][j] *= stiffening_factor
        elif self.precise_stiffness_correction == 0:
            # TODO: check this code, it was 23:52pm
            average_node_volume = self.volume_total/self.nnodes
            # Calculate stiffening factor - surface corrections for 2D/3D problem, for this we need family matrix
            for i in range(0, self.nnodes):
                nnodes_i_family = len(family[i])
                nodei_family_volume = nnodes_i_family * average_node_volume # Possible to calculate more exactly, we have the volumes for free
                for j in range(len(family[i])):
                    nnodes_j_family = len(family[j])
                    nodej_family_volume = nnodes_j_family* average_node_volume # Possible to calculate more exactly, we have the volumes for free
                    
                    stiffening_factor = 2.* self.family_volume /  (nodej_family_volume + nodei_family_volume)
                    
                    bond_stiffness_family[i][j] *= stiffening_factor
        elif self.precise_stiffness_correction == 2:
            # Don't apply stiffness correction factor
            pass

        # Maximum number of nodes that any one of the nodes is connected to, must be a power of 2 (for OpenCL reduction)
        self.max_horizon_length = np.intc(
                    1<<(len(max(family, key=lambda x: len(x)))-1).bit_length()
                )

        self.horizons = -1 * np.ones([self.nnodes, self.max_horizon_length])
        for i, j in enumerate(self.family):
            self.horizons[i][0:len(j)] = j

        self.bond_stiffness = -1. * np.ones([self.nnodes, self.max_horizon_length])
        for i, j in enumerate(self.bond_stiffness_family):
            self.bond_stiffness[i][0:len(j)] = j

        self.bond_critical_stretch = -1. * np.ones([self.nnodes, self.max_horizon_length])
        for i, j in enumerate(self.bond_critical_stretch_family):
            self.bond_critical_stretch[i][0:len(j)] = j

        # Make sure it is in a datatype that C can handle
        self.horizons = self.horizons.astype(np.intc)

        vtk.writeNetwork(self.network_file_name, "Network",
                      self.max_horizon_length, self.horizons_lengths,
                      self.family, self.bond_stiffness_family, self.bond_critical_stretch_family)

    def _increment_load(self, model, load_scale):
        if model.num_force_bc_nodes != 0:
            # update the host force load scale
            self.force_load_scale_h = np.float64(load_scale)
    def _increment_displacement(self, model, displacement_scale):
        # update the host force load scale
        self.displacement_load_scale_h = np.float64(displacement_scale)

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
        :arg int write: The frequency, in number of steps, to write the system
            to a mesh file by calling
            :meth:`peridynamics.model.Model.write_mesh`. If `None` then no
            output is written. Default `None`.
        """
        (u,
         ud,
         damage,
         boundary_function,
         write_path) = self._simulate_initialise(
            integrator, boundary_function, u, connectivity, write_path
            )
             
        # Calculate number of time steps that displacement load is in the 'build-up' phase
        if not ((self.displacement_rate is None) or (self.build_displacement is None) or (self.final_displacement is None)):
            build_time, a, b, c= _calc_build_time(self.build_displacement, self.displacement_rate, steps)

        # Local memory containers for Bond forces
        local_mem_x = cl.LocalMemory(np.dtype(np.float64).itemsize * self.max_horizon_length)
        local_mem_y = cl.LocalMemory(np.dtype(np.float64).itemsize * self.max_horizon_length)
        local_mem_z = cl.LocalMemory(np.dtype(np.float64).itemsize * self.max_horizon_length)

        # Damage vector
        local_mem = cl.LocalMemory(np.dtype(np.float64).itemsize * self.max_horizon_length)

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
                              hostbuf=self.bc_types)
        bc_values_d = cl.Buffer(self.context,
                               cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                               hostbuf=self.bc_values)
        force_bc_types_d = cl.Buffer(self.context,
                              cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                              hostbuf=self.force_bc_types)
        force_bc_values_d = cl.Buffer(self.context,
                               cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                               hostbuf=self.force_bc_values)
        vols_d = cl.Buffer(self.context,
                           cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                           hostbuf=self.V)
        bond_stiffness_d = cl.Buffer(self.context,
                           cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                           hostbuf=np.ascontiguousarray(self.bond_stiffness, dtype=np.float64))
        bond_critical_stretch_d = cl.Buffer(self.context,
                           cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                           hostbuf=np.ascontiguousarray(self.bond_critical_stretch, dtype=np.float64))
        horizons_lengths_d = cl.Buffer(
                self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                hostbuf=self.horizons_lengths)

        # Read and write
        horizons_d = cl.Buffer(
                self.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR,
                hostbuf=self.horizons)
        u_d = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, np.empty((self.nnodes, self.degrees_freedom), dtype=np.float64).nbytes)
        ud_d = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, np.empty((self.nnodes, self.degrees_freedom), dtype=np.float64).nbytes)

        # Write only
        damage_d = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY, np.empty(self.nnodes).astype(np.float64).nbytes)
        # Initialize kernel parameters TODO: no sign of this in Jim's code, is it relevant?
        self.cl_kernel_time_integration.set_scalar_arg_dtypes(
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
             None
             ])
        # Initialize kernel parameters
        self.cl_kernel_update_displacement.set_scalar_arg_dtypes(
            [None,
             None,
             None,
             None,
             None
             ])
        self.cl_kernel_reduce_damage.set_scalar_arg_dtypes(
            [None, None, None, None])

        # Container for plotting data
        damage_sum_data = []
        tip_displacement_data = []
        tip_force_data = []

        # Ease off displacement loading switch
        ease_off = 0
        for step in range(1, steps+1):
            
            # Update displacements
            self._update_displacement(ud_d, u_d, bc_types_d, bc_values_d, displacement_load_scale)
                
            # Calculate the force due to bonds on each node
            self._bond_force(u_d, ud_d, r0_d, vols_d, horizons_d, bond_stiffness_d,
                    bond_critical_stretch_d, force_bc_types_d, force_bc_values_d,
                    local_mem_x, local_mem_y, local_mem_z, force_load_scale,
                    displacement_load_scale)

            if write:
                if step % write == 0:
                    # Calculate the damage
                    self._damage(horizons_d, horizons_lengths_d, damage_d, local_mem)

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
            if not (self.load_scale_rate is None):
                load_scale = min(1.0, self.load_scale_rate * step)
                if load_scale != 1.0:
                    self._increment_load(self, load_scale)
            # Increase dispalcement in 5th order polynomial increments
            if not ((self.displacement_rate is None) or (self.build_displacement is None) or (self.final_displacement is None)):
                # 5th order polynomial/ linear curve used to calculate displacement_scale
                displacement_scale, ease_off = _calc_load_displacement_rate(a, b, c,
                                                                 self.final_displacement,
                                                                 build_time,
                                                                 self.displacement_rate,
                                                                 step, 
                                                                 self.build_displacement,
                                                                 ease_off)
                if displacement_scale != 0.0:
                    self._increment_displacement(displacement_scale)
            # No user specified build up parameters case
            elif not (self.displacement_rate is None):
                self._increment_displacement(1.0)

        return damage_sum_data, tip_displacement_data, tip_force_data

    def _simulate_initialise(self, integrator, boundary_function, u, ud,
                             connectivity, write_path):
        """
        Initialise simulation variables.

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

        return u, ud, damage, boundary_function, write_path

class ContextError(Exception):
    """No suitable context was found by :func:`get_context`."""

    def __init__(self):
        """Exception constructor."""
        message = ("No suitable context was found. You can manually specify"
                   "the context by passing it to ModelCL with the 'context'"
                   "argument.")
        super().__init__(message)

# Utility functions should probabily not live here.
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
    
    return(midpoint_gradient, a, b, c)

def _calc_build_time(build_displacement, displacement_scale_rate, steps):
    T = 0
    midpoint_gradient = np.inf
    while midpoint_gradient > displacement_scale_rate:
        try:
            midpoint_gradient, a, b, c = _calc_midpoint_gradient(T, build_displacement)
        except:
            pass
        T += 1
        if T > steps:
            # TODO: suggest some valid values from the parameters given
            raise ValueError('Displacement build-up time was larger than total simulation time steps! \ntry decreasing build_displacement, or increase max_displacement_rate. steps = {}'.format(steps))
            break
    return(T, a, b, c)

def _calc_load_displacement_rate(a, b, c, final_displacement, build_time, displacement_scale_rate, step, build_displacement, ease_off):
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
    else: # linear regime
        # calculate displacement
        linear_regime_time = step - build_time/2
        linear_regime_displacement = linear_regime_time * displacement_scale_rate
        displacement = linear_regime_displacement + build_displacement/2
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
        
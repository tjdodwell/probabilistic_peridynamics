"""Peridynamics model."""
from .integrators import Integrator
from .utilities import write_array
from .neighbour_list import (set_family, create_neighbour_list_cl,
                             create_neighbour_list_cython, create_crack_cython,
                             create_crack_cl)
from collections import namedtuple
import numpy as np
import pathlib
from tqdm import trange
import warnings
import meshio

_MeshElements = namedtuple("MeshElements", ["connectivity", "boundary"])
_mesh_elements_2d = _MeshElements(connectivity="triangle",
                                  boundary="line")
_mesh_elements_3d = _MeshElements(connectivity="tetra",
                                  boundary="triangle")


class Model(object):
    """
    A peridynamics model.

    This class allows users to define a composite, non-linear peridynamics
    system from parameters and a set of initial conditions (coordinates,
    connectivity and optionally bond_types and stiffness_corrections). For
    this an :class:`peridynamics.integrators.Integrator` is required, and
    optionally functions implementing the boundarys.

        >>> from peridynamics import Model
        >>> from peridynamics.integrators import Euler
        >>>
        >>> def is_displacement_boundary(x):
        >>>     # Particle does not live on a boundary
        >>>     bnd = [None, None, None]
        >>>     # Particle does live on a boundary
        >>>     if x[0] < 1.5 * 0.1:
        >>>         # Displacements BCs are applied in negative x direction
        >>>         bnd[0] = -1
        >>>     elif x[0] > 1.0 - 1.5 * 0.1:
        >>>         # Displacement BCs are applied in positive x direction
        >>>         bnd[0] = 1
        >>>     return bnd
        >>>
        >>> model = Model(
        >>>     mesh_file,
        >>>     integrator=euler,
        >>>     horizon=0.1,
        >>>     critical_stretch=0.005,
        >>>     bond_stiffness=18.00 * 0.05 / (np.pi * 0.1**4),
        >>>     is_displacement_boundary=is_displacement_boundary,
        >>>     )

    To define a crack in the inital configuration, you may supply a list of
    pairs of particles between which the crack is.

        >>> initial_crack = [(1,2), (5,7), (3,9)]
        >>> model = Model(
        >>>     mesh_file,
        >>>     integrator=euler,
        >>>     horizon=0.1,
        >>>     critical_stretch=0.005,
        >>>     bond_stiffness=18.00 * 0.05 / (np.pi * 0.1**4),
        >>>     is_displacement_boundary=is_displacement_boundary,
        >>>     initial_crack=initial_crack
        >>>     )

    If it is more convenient to define the crack as a function you may also
    pass a function to the constructor which takes the array of coordinates as
    its only argument and returns a list of tuples as described above. The
    :func:`peridynamics.model.initial_crack_helper` decorator has been provided
    to easily create a function of the correct form from one which tests a
    single pair of node coordinates and returns `True` or `False`.

        >>> from peridynamics import initial_crack_helper
        >>>
        >>> @initial_crack_helper
        >>> def initial_crack(x, y):
        >>>     ...
        >>>     if crack:
        >>>         return True
        >>>     else:
        >>>         return False
        >>>
        >>> model = Model(
        >>>     mesh_file,
        >>>     integrator=euler,
        >>>     horizon=0.1,
        >>>     critical_stretch=0.005,
        >>>     bond_stiffness=18.00 * 0.05 / (np.pi * 0.1**4),
        >>>     is_displacement_boundary=is_displacement_boundary,
        >>>     initial_crack=initial_crack
        >>>     )

    The :meth:`Model.simulate` method can be used to conduct a peridynamics
    simulation. Here it is possible to define the boundary conditions

        >>> model = Model(...)
        >>>
        >>>
        >>> u, damage, *_ = model.simulate(
        >>>     steps=1000,
        >>>     max_displacement_rate=0.000005/2,
        >>>     write=100
        >>>     )
    """

    def __init__(self, mesh_file, integrator, horizon, critical_stretch,
                 bond_stiffness, transfinite=0,
                 volume_total=None, write_path=None, connectivity=None,
                 family=None, volume=None, initial_crack=[], dimensions=2,
                 is_density=None, is_bond_type=None,
                 is_displacement_boundary=None, is_forces_boundary=None,
                 is_tip=None, density=None, bond_types=None,
                 stiffness_corrections=None,
                 precise_stiffness_correction=None):
        """
        Create a :class:`ModelCLBen` object.

        :arg str mesh_file: Path of the mesh file defining the systems nodes
            and connectivity.
        :arg  integrator: The integrator to use, see
            :mod:`peridynamics.integrators` for options.
        :type integrator: :class:`peridynamics.integrators.Integrator`
        :arg float horizon: The horizon radius. Nodes within `horizon` of
            another interact with that node and are said to be within its
            neighbourhood.
        :arg critical_stretch: An (nregimes, nbond_types) array of critical
            stretch values, each corresponding to a bond type and a regime,
            or a float value of the critical stretch of the Peridynamic
            bond-based prototype microelastic brittle (PMB) model.
        :type critical_stretch: :class:`numpy.ndarray` or float
        :arg bond_stiffness: An (nregimes, nbond_types) array of bond
            stiffness values, each corresponding to a bond type and a regime,
            or a float value of the bond stiffness the Peridynamic bond-based
            prototype microelastic brittle (PMB) model.
        :type bond_stiffness: :class:`numpy.ndarray` or float
        :arg bool transfinite: Set to 1 for Cartesian cubic (tensor grid) mesh.
            Set to 0 for a tetrahedral mesh (default). If set to 1, the
            volumes of the nodes are approximated as the average volume of
            nodes on a cuboidal tensor-grid mesh.
        :arg float volume_total: Total volume of the mesh. Must be provided if
            transfinite mode (transfinite=1) is used.
        :arg write_path: The path where the model arrays, (volume, family,
            connectivity, stiffness_corrections, bond_types) should be
            written to file to avoid overhead.
        :type write_path: path-like or str
        :arg connectivity: The initial connectivity for the model. A tuple
            of a neighbour list and the number of neighbours for each node. If
            `None` the connectivity at the time of construction of the
            :class:`Model` object will be used. Default `None`.
        :type connectivity: tuple(:class:`numpy.ndarray`,
            :class:`numpy.ndarray`)
        :arg family: The family array. An array of the intial number of nodes
            within the horizon of each node. If `None` the family at the
            time of construction of the :class:`Model` object will be used.
            Default `None`.
        :type family: :class:`numpy.ndarray`
        :arg volume: Array of volumes for each node. If `None` the volume
            at the time of construction of the :class:`Model` object will be
            used. Default `None`.
        :type volume: :class:`numpy.ndarray`
        :arg initial_crack: The initial crack of the system. The argument may
            be a list of tuples where each tuple is a pair of integers
            representing nodes between which to create a crack. Alternatively,
            the arugment may be a function which takes the (nnodes, 3)
            :class:`numpy.ndarray` of coordinates as an argument, and returns a
            list of tuples defining the initial crack. Default is []
        :type initial_crack: list(tuple(int, int)) or function
        :arg int dimensions: The dimensionality of the model. The
            default is 2.
        :arg is_density: A function that returns a float of the material
            density, given a node coordinate as input.
        :type is_density: function
        :arg is_bond_type: A function that returns an integer value (a
            flag) of the bond_type, given two node coordinates as input.
        :type is_bond_type: function
        :arg is_displacement_boundary: A function to determine if a node is on
            the boundary for a displacement boundary condition, and if it is,
            which direction and magnitude the boundary conditions are applied
            (positive or negative cartesian direction). It has the form
            is_displacement_boundary(:class:`numpy.ndarray`). The argument is
            the initial coordinates of a particle being simulated.
            `is_displacement_boundary` returns a (3) list of the boundary types
            in each cartesian direction.
            A boundary type with an int value of None if the particle is not
            on a displacement controlled boundary, a value of 1 if is is on a
            boundary and loaded in the positive cartesian direction, and a
            value of -1 if it is on the boundary and loaded in the negative
            direction, and a value of 0 if it is not loaded.
        :type is_displacement_boundary: function
        :arg is_forces_boundary: As 'is_displacement_boundary' but applying to
            force boundary conditions as opposed to displacement boundary
            conditions.
        :type is_forces_boundary: function
        :arg is_tip: A function to determine if a node is to be measured for
            its reaction force or displacement over time, and if it is, which
            direction the measurements are made
            (positive or negative cartesian direction). It has the form
            is_tip(:class:`numpy.ndarray`). The argument is the initial
            coordinates of a particle being simulated. `is_tip` returns a
            (3) list of the measurement types in each cartesian direction.
            A boundary type with an int value of None if the particle is not on
            the `tip` to be measured, a value of 1 if is is on the `tip` and
            to be measured.
        :type is_tip: function
        :arg density: An (nnodes, ) array of node density values, each
            corresponding to a material
        :type density: :class:`numpy.ndarray`
        :arg bond_types: The bond_types for the model.
            If `None` the bond_types at the time of construction of the
            :class:`Model` object will be used. Default `None`.
        :type bond_types: :class:`numpy.ndarray`
        :arg stiffness_corrections: The stiffness_corrections for
            the model. If `None` the stiffness_corrections at the time
            of construction of the :class:`Model` object will be used. Default
            `None`.
        :type stiffness_corrections: :class:`numpy.ndarray`
        :arg int precise_stiffness_correction: A switch variable. Set to 1:
            Stiffness corrections are calculated more accurately using
            actual nodal volumes. Set to 0: Stiffness corrections are calculate
            using an average nodal volume. Set to None: All stiffness
            corrections are set to 1.0, i.e. no stiffness correction is
            applied.

        :raises DimensionalityError: when an invalid `dimensions` argument is
            provided.
        :raises FamilyError: when a node has no neighbours (other nodes it
            interacts with) in the initial state.

        :returns: A new :class:`Model` object.
        :rtype: Model
        """
        if not isinstance(integrator, Integrator):
            raise InvalidIntegrator(integrator)
        else:
            self.integrator = integrator

        # If no write path was provided, assign it as None so that model arrays
        # are not written, otherwise, ensure write_path is a Path objects
        if write_path is None:
            self.write_path = None
        else:
            self.write_path = pathlib.Path(write_path)

        # Set model dimensionality
        self.dimensions = dimensions

        if dimensions == 2:
            self.mesh_elements = _mesh_elements_2d
        elif dimensions == 3:
            self.mesh_elements = _mesh_elements_3d
        else:
            raise DimensionalityError(dimensions)

        # Read coordinates and connectivity from mesh file
        self._read_mesh(mesh_file)

        self.horizon = horizon

        # Calculate the volume for each node, if None is provided
        if volume is None:
            # Calculate the volume for each node
            if self.nnodes > 70000:
                warnings.warn(
                        "Calculating volume... this may take a while.")
            self.volume, self.sum_total_volume = self._volume(
                transfinite, volume_total)
            if write_path is not None:
                write_array(write_path, "volume", self.volume)
        elif type(volume) == np.ndarray:
            if np.shape(volume) != (self.nnodes, ):
                raise ValueError("volume shape is wrong, and must be "
                                 "(nnodes, ) (expected {}, got {})".format(
                                     (self.nnodes, ),
                                     np.shape(volume)))
            warnings.warn(
                    "Reading volume from argument.")
            self.volume = volume.astype(np.float64)
        else:
            raise TypeError("volume type is wrong (expected {}, got "
                            "{})".format(type(volume),
                                         np.ndarray))

        # Calculate the family (number of bonds in the initial configuration)
        # for each node, if None is provided
        if family is None:
            # Calculate family
            if self.nnodes > 70000:
                warnings.warn(
                        "Calculating family... this may take a while.")
            self.family = set_family(self.coords, horizon)
            if write_path is not None:
                write_array(write_path, "family", self.family)
        elif type(family) == np.ndarray:
            if np.shape(family) != (self.nnodes, ):
                raise ValueError("family shape is wrong, and must be "
                                 "(nnodes, ) (expected {}, got {})".format(
                                     (self.nnodes, ),
                                     np.shape(family)))
            warnings.warn(
                    "Reading family from argument.")
            self.family = family.astype(np.intc)
        else:
            raise TypeError("family type is wrong (expected {}, got "
                            "{})".format(type(family),
                                         np.ndarray))
        if np.any(self.family == 0):
            raise FamilyError(self.family)

        if integrator.context is None:
            if connectivity is None:
                # Create the neighbourlist for the cython implementation
                if self.nnodes > 70000:
                    warnings.warn(
                        "Calculating connectivity... this may take a while.")
                self.max_neighbours = self.family.max()
                nlist, n_neigh = create_neighbour_list_cython(
                    self.coords, horizon, self.max_neighbours
                    )
            elif type(connectivity) == tuple:
                if len(connectivity) != 2:
                    raise ValueError("connectivity size is wrong (expected 2,"
                                     " got {})".format(len(connectivity)))
                warnings.warn(
                    "Reading connectivity from argument.")
                nlist, n_neigh = connectivity
                nlist = nlist.astype(np.intc)
                n_neigh = n_neigh.astype(np.intc)
                self.max_neighbours = np.intc(
                            np.shape(nlist)[1]
                        )
                if self.max_neighbours != self.family.max():
                    raise ValueError(
                        "max_neighbours, which is equal to the"
                        " size of axis 1 of nlist is wrong (expected "
                        " max_neighbours = np.shape(nlist)[1] = family.max()"
                        " = {}, got {})".format(
                            self.family.max(), self.max_neighbours))
            else:
                raise TypeError("connectivity type is wrong (expected {} or"
                                " {}, got {})".format(
                                    tuple, type(None), type(connectivity)))
            # Initialise initial crack for cython
            if initial_crack:
                if callable(initial_crack):
                    initial_crack = initial_crack(
                        self.coords, nlist, n_neigh)
                create_crack_cython(
                    np.array(initial_crack, dtype=np.int32),
                    nlist, n_neigh
                    )

        else:
            if connectivity is None:
                # Create the neighbourlist for the OpenCL implementation
                if self.nnodes > 70000:
                    warnings.warn(
                        "Calculating connectivity... this may take a while.")
                self.max_neighbours = np.intc(
                            1 << (int(self.family.max() - 1)).bit_length()
                        )
                nlist, n_neigh = create_neighbour_list_cl(
                    self.coords, horizon, self.max_neighbours
                    )
                if write_path is not None:
                    write_array(self.write_path, "nlist", nlist)
                    write_array(self.write_path, "n_neigh", n_neigh)
            elif type(connectivity) == tuple:
                if len(connectivity) != 2:
                    raise ValueError("connectivity size is wrong (expected 2, "
                                     " got {})".format(len(connectivity)))
                warnings.warn(
                    "Reading connectivity from argument.")
                nlist, n_neigh = connectivity
                nlist = nlist.astype(np.intc)
                n_neigh = n_neigh.astype(np.intc)
                self.max_neighbours = np.intc(
                            np.shape(nlist)[1]
                        )
                test = self.max_neighbours - 1
                if self.max_neighbours & test:
                    raise ValueError(
                        "max_neighbours, which is equal to the"
                        " size of axis 1 of nlist is wrong (expected "
                        " max_neighbours = np.shape(nlist)[1] = {},"
                        " got {})".format(
                            1 << (int(self.family.max() - 1)).bit_length(),
                            self.max_neighbours))
            else:
                raise TypeError("connectivity type is wrong (expected {} or"
                                " {}, got {})".format(
                                    tuple, type(None), type(connectivity)))
            # Initialise initial crack for OpenCL
            if initial_crack:
                if callable(initial_crack):
                    initial_crack = initial_crack(
                        self.coords, nlist, n_neigh)
                create_crack_cl(
                    np.array(initial_crack, dtype=np.int32),
                    nlist, n_neigh, self.family
                    )

        self.initial_connectivity = (nlist, n_neigh)
        self.degrees_freedom = 3

        if stiffness_corrections is None:
            if precise_stiffness_correction is None:
                self.stiffness_corrections = None
            else:
                # Calculate stiffness correction factors and write to file
                self.stiffness_corrections = self._set_stiffness_corrections(
                    self.horizon, self.initial_connectivity,
                    precise_stiffness_correction, self.write_path)
        elif type(stiffness_corrections) == np.ndarray:
            if np.shape(stiffness_corrections) != (
                    self.nnodes, self.max_neighbours):
                raise ValueError("stiffness_corrections shape is wrong, "
                                 "and must be (nnodes, max_neighbours) "
                                 "(expected {}, got {})".format(
                                     (self.nnodes, self.max_neighbours),
                                     np.shape(stiffness_corrections)))
            else:
                warnings.warn(
                    "Reading stiffness_corrections from argument.")
                self.stiffness_corrections = (
                    stiffness_corrections.astype(np.float64))
        else:
            raise TypeError("stiffness_corrections type is wrong (expected {}"
                            ", got {})".format(
                                    np.ndarray, type(stiffness_corrections)))

        # Create dummy is_bond_type function is none is provided
        if is_bond_type is None:
            def is_bond_type(x, y):
                return 0

        # Set damage model
        (self.bond_stiffness,
         self.critical_stretch,
         self.plus_cs,
         self.nbond_types,
         self.nregimes) = self._set_damage_model(
             bond_stiffness, critical_stretch)

        if bond_types is None:
            # Calculate bond types and write to file
            bond_types = self._set_bond_types(
                self.initial_connectivity, is_bond_type,
                self.nbond_types, self.nregimes, self.write_path)

        elif type(bond_types) == np.ndarray:
            if np.shape(bond_types) != (self.nnodes, self.max_neighbours):
                raise ValueError("bond_types shape is wrong, "
                                 "and must be (nnodes, max_neighbours) "
                                 "(expected {}, got {})".format(
                                     (self.nnodes, self.max_neighbours),
                                     np.shape(bond_types)))
            warnings.warn(
                "Reading bond_types from argument.")
            bond_types = bond_types.astype(np.intc)
        else:
            raise TypeError("bond_types type is wrong (expected {}"
                            ", got {})".format(
                                    np.ndarray, type(bond_types)))

        # Set densities of the model
        self.densities = self._set_densities(
            density, is_density, self.write_path)

        # Create dummy boundary conditions functions if none is provided
        if is_forces_boundary is None:
            def is_forces_boundary(x):
                # Particle does not live on forces boundary
                bnd = [None, None, None]
                return bnd
        if is_displacement_boundary is None:
            def is_displacement_boundary(x):
                # Particle does not live on displacement boundary
                bnd = [None, None, None]
                return bnd
        if is_tip is None:
            def is_tip(x):
                # Particle does not live on tip
                bnd = [None, None, None]
                return bnd

        # Apply boundary conditions
        (bc_types,
         bc_values,
         force_bc_types,
         force_bc_values,
         self.tip_types) = self._set_boundary_conditions(
            is_displacement_boundary, is_forces_boundary, is_tip)

        # Build the integrator
        self.integrator.build(
            self.nnodes, self.degrees_freedom, self.max_neighbours,
            self.coords, self.volume, self.family, bc_types, bc_values,
            force_bc_types, force_bc_values, self.stiffness_corrections,
            bond_types, self.densities)

    def _read_mesh(self, filename):
        """
        Read the model's nodes, connectivity and boundary from a mesh file.

        :arg str filename: Path of the mesh file to read

        :returns: None
        :rtype: NoneType
        """
        mesh = meshio.read(filename)

        # Get coordinates, encoded as mesh points
        self.coords = np.array(mesh.points, dtype=np.float64)
        self.nnodes = self.coords.shape[0]

        # Get connectivity, mesh triangle cells
        self.mesh_connectivity = mesh.cells_dict[
            self.mesh_elements.connectivity
            ]

        # Get boundary connectivity, mesh lines
        self.mesh_boundary = mesh.cells_dict[self.mesh_elements.boundary]

    def write_mesh(self, filename, damage=None, displacements=None,
                   file_format=None):
        """
        Write the model's nodes, connectivity and boundary to a mesh file.

        :arg str filename: Path of the file to write the mesh to.
        :arg damage: The damage of each node. Default is None.
        :type damage: :class:`numpy.ndarray`
        :arg displacements: An array with shape (nnodes, dim) where each row is
            the displacement of a node. Default is None.
        :type displacements: :class:`numpy.ndarray`
        :arg str file_format: The file format of the mesh file to
            write. Inferred from `filename` if None. Default is None.

        :returns: None
        :rtype: NoneType
        """
        meshio.write_points_cells(
            filename,
            points=self.coords,
            cells=[
                (self.mesh_elements.connectivity, self.mesh_connectivity),
                (self.mesh_elements.boundary, self.mesh_boundary)
                ],
            point_data={
                "damage": damage,
                "displacements": displacements
                },
            file_format=file_format
            )

    def _volume(self, transfinite, volume_total):
        """
        Calculate the value of each node.

        :arg bool transfinite: Set to 1 for Cartesian cubic (tensor grid) mesh.
            Set to 0 for a tetrahedral mesh (default). If set to 1, the
            volumes of the nodes are approximated as the average volume of
            nodes on a cuboidal tensor-grid mesh.
        :arg float volume_total: User input for the total volume of the mesh,
            for checking the sum total of elemental volumes is equal to user
            input volume for simple prismatic problems. In the case where no
            expected total volume is provided, the check is not done.

        :returns: Tuple containing an array of volumes for each node and the
            sum total of all the nodal volumes, which is equal to the total
            mesh volume.
        :rtype: tuple(:class:`numpy.ndarray`, float)
        """
        if transfinite:
            if volume_total is None:
                raise TypeError("In transfinite mode, a total mesh volume "
                                "volume_total' must be provided as a keyword"
                                " argument (expected {}, got {})".format(
                                     float, type(volume_total)))
        volume = np.zeros(self.nnodes)
        dimensions = self.dimensions
        sum_total_volume = 0.0

        if transfinite:
            tmp = volume_total / self.nnodes
            volume = tmp * np.ones(self.nnodes)
            sum_total_volume = volume_total
        else:
            if dimensions == 2:
                # element is a triangle
                element_nodes = 3
            elif dimensions == 3:
                # element is a tetrahedron
                element_nodes = 4

            for nodes in self.mesh_connectivity:
                # Calculate volume/area or element
                if dimensions == 2:
                    a, b, c = self.coords[nodes]

                    # Area of a trianble
                    i = b - a
                    j = c - a
                    element_volume = 0.5 * np.linalg.norm(np.cross(i, j))
                    sum_total_volume += element_volume
                elif dimensions == 3:
                    a, b, c, d = self.coords[nodes]

                    # Volume of a tetrahedron
                    i = a - d
                    j = b - d
                    k = c - d
                    element_volume = abs(np.dot(i, np.cross(j, k))) / 6
                    sum_total_volume += element_volume

                # Add fraction element volume to all nodes belonging to that
                # element
                volume[nodes] += element_volume / element_nodes
        volume = volume.astype(np.float64)
        sum_total_volume = np.float64(sum_total_volume)
        return (volume, sum_total_volume)

    def _set_densities(self, density, is_density, write_path):
        """
        Build densities array.

        :arg density: An (nnodes, ) array of node density values, each
            corresponding to a material, or a float value of the density
            if nmaterials=1.
        :type density: :class:`numpy.ndarray`
        :arg is_density: A function that returns a float of the material
            density, given a node coordinate as input.
        :type is_density: function
        :arg write_path: The path where the vtk files should be written.
        :type write_path: path-like or str

        :returns: A (nnodes, degrees_freedom) array of nodal densities, or
            None if no is_density function or density array is supplied.
        :rtype: :class:`numpy.ndarray` or None
        """
        if density is None:
            if is_density is None:
                densities = None
            else:
                if not callable(is_density):
                    raise TypeError(
                        "is_density must be a *function*.")
                elif type(is_density(self.coords[0])) is not float:
                    raise TypeError(
                        "is_density must be a function that returns a *float* "
                        "(expected {}, got {})".format(
                            float, type(
                                is_density(self.coords[0]))))
                density = np.ones(self.nnodes)
                for i in range(self.nnodes):
                    density[i] = is_density(self.coords[i])
                if write_path is not None:
                    write_array(write_path, "density", density)
                densities = np.transpose(
                    np.tile(density, (self.degrees_freedom, 1))).astype(
                        np.float64)
        elif type(density) == np.ndarray:
            if np.shape(density) != (self.nnodes,):
                raise ValueError("densty shape is wrong, and must be "
                                 "(nnodes,) (expected {}, got {})".format(
                                     (self.nnodes,), np.shape(density)))
            densities = np.transpose(
                np.tile(density, (self.degrees_freedom, 1))).astype(np.float64)
        else:
            raise TypeError("density type is wrong, and must be an array of"
                            " shape (nnodes,) (expected {}, got {})".format(
                                np.ndarray, type(density)))
        return densities

    def _set_bond_types(self, connectivity, is_bond_type, nbond_types,
                        nregimes, write_path):
        """
        Build bond_types array.

        :arg connectivity: The initial connectivity for the simulation. A tuple
            of a neighbour list and the number of neighbours for each node. If
            `None` the connectivity at the time of construction of the
            :class:`Model` object will be used. Default `None`.
        :type connectivity: tuple(:class:`numpy.ndarray`,
            :class:`numpy.ndarray`)
        :arg is_bond_type: A function that returns an integer value (a
            flag) of the bond type, given two node coordinates as input.
        :type is_bond_type: function
        :arg int nbond_types: The expected number of damage models, one
            for each type of bond.
        :arg int nbond_types: The expected number of regimes in the damage
            model.
        :arg write_path: The path where the vtk files should be written.
        :type write_path: path-like or str

        :returns: A (`nnodes`, `max_neighbours`) array of the bond types,
            which are used to index into the bond_stiffness and
            critical_stretch arrays.
        :rtype: :class:`numpy.ndarray`
        """
        if nbond_types != 1:
            if not callable(is_bond_type):
                raise TypeError(
                    "is_bond_type must be a *function*.")
            if type(is_bond_type(self.coords[0], self.coords[1])) is not int:
                raise TypeError(
                    "is_bond_type must be a function that returns an *int* "
                    "(expected {}, got {})".format(
                        int, type(
                            is_bond_type(self.coords[0], self.coords[1]))))
            nlist, n_neigh = connectivity
            bond_types = np.zeros(
                (self.nnodes, self.max_neighbours))
            for i in range(self.nnodes):
                for neigh in range(n_neigh[i]):
                    j = nlist[i][neigh]
                    bond_types[i][neigh] = is_bond_type(
                        self.coords[i, :], self.coords[j, :])
            bond_types = bond_types.astype(np.intc)
            if np.any(bond_types > (nbond_types-1)):
                raise ValueError("number of bond types must be equal to the"
                                 " number of bond_stiffness values (expected"
                                 " {}, got {})".format(
                                     nbond_types, np.max(bond_types)))
            if write_path is not None:
                write_array(write_path, "bond_types", bond_types)
        elif nregimes != 1:
            bond_types = np.zeros(
                (self.nnodes, self.max_neighbours))
            bond_types = bond_types.astype(np.intc)
            if write_path is not None:
                write_array(write_path, "bond_types", bond_types)
        else:
            bond_types = None
        return bond_types

    def _set_stiffness_corrections(
            self, horizon, connectivity, precise_stiffness_correction,
            write_path):
        """
        Calculate an array of stiffness correction factors.

        Calculates stiffness correction factors that reduce the peridynamics
        surface softening effect for 2D/3D problem and writes to file.
        The 'volume method' proposed in Chapter 2 in Bobaru F, Foster JT,
        Geubelle PH, Silling SA (2017) Handbook of peridynamic modeling
        (p51 – 52) is used when precise_stiffness_correction = 1 or 0.

        :arg float horizon: The horizon distance.
        :arg connectivity: The initial connectivity for the simulation. A tuple
            of a neighbour list and the number of neighbours for each node. If
            `None` the connectivity at the time of construction of the
            :class:`Model` object will be used. Default `None`.
        :type connectivity: tuple(:class:`numpy.ndarray`,
            :class:`numpy.ndarray`)
        :arg int precise_stiffness_correction: A flag variable. Set to 1:
            Stiffness corrections are calculated more accurately using
            actual nodal volumes. Set to 0: Stiffness corrections are calculate
            using an average nodal volume. Set to None: All stiffness
            corrections are set to 1.0, i.e. no stiffness correction is
            applied.
        :arg write_path: The path where the vtk files should be written.
        :type write_path: path-like or str

        :returns: An (`nnodes`, `max_neighbours`) array of the stiffness
            correction factor of each bond for each node.
        :rtype: :class:`numpy.ndarray`
        """
        nlist, n_neigh = connectivity
        stiffness_corrections = np.ones((self.nnodes, self.max_neighbours))

        if self.dimensions == 2:
            family_volume_bulk = np.pi*np.power(horizon, 2)
        elif self.dimensions == 3:
            family_volume_bulk = (4./3)*np.pi*np.power(horizon, 3)

        if precise_stiffness_correction == 1:
            family_volumes = np.zeros(self.nnodes)
            for i in range(0, self.nnodes):
                tmp = 0.0
                neighbour_list = nlist[i][:self.family[i]]
                for j in range(self.family[i]):
                    tmp += self.volume[neighbour_list[j]]
                family_volumes[i] = tmp
            for i in range(0, self.nnodes):
                family_volume_i = family_volumes[i]
                for neigh in range(n_neigh[i]):
                    family_volume_j = family_volumes[nlist[i][neigh]]
                    stiffness_correction_factor = 2. * family_volume_bulk / (
                        family_volume_i + family_volume_j)
                    stiffness_corrections[i][neigh] = (
                        stiffness_correction_factor)

        elif precise_stiffness_correction == 0:
            average_node_volume = self.sum_total_volume / self.nnodes
            for i in range(0, self.nnodes):
                nnodes_i_family = n_neigh[i]
                nodei_family_volume = nnodes_i_family * average_node_volume
                for neigh in range(nnodes_i_family):
                    j = nlist[i][neigh]
                    nnodes_j_family = n_neigh[j]
                    nodej_family_volume = nnodes_j_family * average_node_volume
                    stiffness_correction_factor = 2. * family_volume_bulk / (
                        nodej_family_volume + nodei_family_volume)
                    stiffness_corrections[i][neigh] = (
                        stiffness_correction_factor)
        else:
            raise ValueError("precise_stiffness_correction value is wrong "
                             "(expected 0 or 1 or None, got {})".format(
                                 precise_stiffness_correction))
        stiffness_corrections = stiffness_corrections.astype(np.float64)
        if write_path is not None:
            write_array(
                write_path,
                "stiffness_corrections", stiffness_corrections)
        return stiffness_corrections

    def _set_damage_model(self, bond_stiffness, critical_stretch):
        """
        Calculate the parameters for the damage models.

        Calculates the `+ c`s (c.f. `y = mx + c`) for the n-linear
        damage models for each bond type, where n is nregimes, e.g. linear,
        bi-linear, tri-linear, etc. from the bond_stiffness and
        critical_stretch values provided.

        :arg bond_stiffness: An (nregimes, nbond_types) array of bond
            stiffness values, each corresponding to a bond type and a regime.
        :type bond_stiffness: list or :class:`numpy.ndarray`
        :arg critical_stretch: An (n_regimes, nbond_types) array of critical
            stretch values, each corresponding to a bond type and a regime.
        :type critical_stretch: list or :class:`numpy.ndarray`
        :arg int n_regimes:

        :returns: A tuple of the damage model:
            bond_stiffness, a float or array of the bond stiffness(es);
            critcal_stretch, a float or array of the critical stretch(es);
            nbond_types, the number of bond types (damage models) in the model;
            nregimes, the number of `regimes` in the damage model. e.g.
                linear has n_regimes = 1, bi-linear has n_regimes = 2, etc;
            plus_cs, an (`nregimes`, `nbond_types`) array of the `+cs` for each
                linear part of the bond damage models for each bond type. Takes
                a value of None if the PMB (Prototype Micro-elastic Brittle)
                model is used, i.e. n_regimes = 1.
        :rtype: tuple(:class:`numpy.ndarray` or :class:`numpy.float64`,
                      :class:`numpy.ndarray` or :class:`numpy.float64`,
                      :class:`numpy.intc`,
                      :class:`numpy.intc`,
                      :class:`numpy.ndarray` or NoneType)
        """
        if type(bond_stiffness) != type(critical_stretch):
            raise TypeError(
                "bond_stiffness must be the same type "
                "as critical_stretch (expected {}, got {})".format(
                    type(critical_stretch),
                    type(bond_stiffness)))
        if ((type(bond_stiffness) is list) or
                (type(bond_stiffness) is np.ndarray)):
            if np.shape(bond_stiffness) != np.shape(critical_stretch):
                raise ValueError(
                    "The shape of bond_stiffness must be equal to the shape "
                    "of critical_stretch (expected {}, got {})".format(
                        np.shape(critical_stretch),
                        np.shape(bond_stiffness)))
            else:
                if np.shape(bond_stiffness) == (1,):
                    nregimes = 1
                    nbond_types = 1
                    bond_stiffness = np.float64(bond_stiffness[0])
                    critical_stretch = np.float64(critical_stretch[0])
                    plus_cs = None
                elif np.shape(bond_stiffness) == ():
                    nregimes = 1
                    nbond_types = 1
                    bond_stiffness = np.float64(bond_stiffness)
                    critical_stretch = np.array(critical_stretch)
                    plus_cs = None
                elif np.shape(bond_stiffness[0]) == (1,):
                    nregimes = 1
                    nbond_types = np.shape(bond_stiffness)[0]
                    bond_stiffness = np.array(
                        bond_stiffness, dtype=np.float64)
                    critical_stretch = np.array(
                        critical_stretch, dtype=np.float64)
                    plus_cs = np.zeros(nbond_types)
                    plus_cs = plus_cs.astype(np.float64)
                elif np.shape(bond_stiffness[0]) == ():
                    nbond_types = 1
                    nregimes = np.shape(bond_stiffness)[0]
                    bond_stiffness = np.array(
                        bond_stiffness, dtype=np.float64)
                    critical_stretch = np.array(
                        critical_stretch, dtype=np.float64)
                    plus_cs = np.zeros(nregimes)
                    c_i = 0.0
                    # The bond force density at 0 stretch is 0
                    plus_cs[0] = c_i
                    c_prev = c_i
                    for r in range(1, nregimes):
                        c_i = (
                            c_prev
                            + (bond_stiffness[r - 1] *
                                critical_stretch[r - 1])
                            - (bond_stiffness[r] *
                               critical_stretch[r - 1]))
                        plus_cs[r] = c_i
                        c_prev = c_i
                    plus_cs = plus_cs.astype(np.float64)
                else:
                    nregimes = np.shape(bond_stiffness)[1]
                    nbond_types = np.shape(bond_stiffness)[0]
                    bond_stiffness = np.array(
                        bond_stiffness, dtype=np.float64)
                    critical_stretch = np.array(
                        critical_stretch, dtype=np.float64)
                    plus_cs = np.zeros((nbond_types, nregimes))
                    c_i = np.zeros(nbond_types)
                    # The bond force density at 0 stretch is 0
                    plus_cs[:, 0] = c_i
                    c_prev = c_i
                    for r in range(1, nregimes):
                        c_i = (
                            c_prev
                            + (bond_stiffness[:, r - 1] *
                                critical_stretch[:, r - 1])
                            - (bond_stiffness[:, r] *
                               critical_stretch[:, r - 1]))
                        plus_cs[:, r] = c_i
                        c_prev = c_i
                    plus_cs = plus_cs.astype(np.float64)
        elif ((type(bond_stiffness) is float) or
              (type(bond_stiffness) is np.float64)):
            nregimes = 1
            nbond_types = 1
            bond_stiffness = np.float64(bond_stiffness)
            critical_stretch = np.float64(critical_stretch)
            plus_cs = None
        else:
            raise TypeError(
                "Type of bond_stiffness and critical_stretch is not supported "
                "(expected {} or {}, got {})".format(
                    float, np.ndarray, type(bond_stiffness)))

        # Convert to types that OpenCL can handle
        nregimes = np.intc(nregimes)
        nbond_types = np.intc(nbond_types)
        if np.any(critical_stretch < 0):
            raise ValueError("critical_stretch values must not be < 0, "
                             "(got {})".format(critical_stretch))
        return (bond_stiffness, critical_stretch, plus_cs, nbond_types,
                nregimes)

    def _set_boundary_conditions(
            self, is_displacement_boundary, is_forces_boundary, is_tip):
        """
        Set the boundary conditions of the model.

        :arg is_displacement_boundary: A function to determine if a node is on
            the boundary for a displacement boundary condition, and if it is,
            which direction and magnitude the boundary conditions are applied
            (positive or negative cartesian direction). It has the form
            is_displacement_boundary(:class:`numpy.ndarray`). The argument is
            the initial coordinates of a particle being simulated.
            `is_displacement_boundary` returns a (3) list of the boundary types
            in each cartesian direction.
            A boundary type with an int value of None if the particle is not
            on a displacement controlled boundary, a value of 1 if is is on a
            boundary and loaded in the positive cartesian direction, and a
            value of -1 if it is on the boundary and loaded in the negative
            direction, and a value of 0 if it is not loaded.
        :type is_displacement_boundary: function
        :arg is_forces_boundary: As 'is_displacement_boundary' but applying to
            force boundary conditions as opposed to displacement boundary
            conditions.
        :type is_forces_boundary: function
        :arg is_tip: A function to determine if a node is to be measured for
            its reaction force or displacement over time, and if it is, which
            direction the measurements are made
            (positive or negative cartesian direction). It has the form
            is_tip(:class:`numpy.ndarray`). The argument is the initial
            coordinates of a particle being simulated. `is_tip` returns a
            (3) list of the measurement types in each cartesian direction.
            A boundary type with an int value of None if the particle is not on
            the `tip` to be measured, a value of 1 if is is on the `tip` and
            to be measured.
        :type is_tip: function

        :returns: A tuple of the displacement and foce boundary condition types
            and values, and the tip types.
        :rtype: tuple(:class:`numpy.ndarray`, :class:`numpy.ndarray`,
                      :class:`numpy.ndarray`, :class:`numpy.ndarray`)
        """
        functions = {'is_displacement_boundary': is_displacement_boundary,
                     'is_forces_boundary': is_forces_boundary,
                     'is_tip': is_tip}
        for function in functions:
            if not callable(functions[function]):
                raise TypeError("{} must be a *function*.".format(function))
            if type(functions[function]([0, 0, 0])) is not list:
                raise TypeError(
                    "{} must be a function that returns a *list*.".format(
                        function))
            if len(functions[function]([0, 0, 0])) != 3:
                raise TypeError("{} must return a function that returns a list"
                                " of length *3* of floats or None")

        bc_types = np.zeros(
            (self.nnodes, self.degrees_freedom), dtype=np.intc)
        bc_values = np.zeros(
            (self.nnodes, self.degrees_freedom), dtype=np.float64)
        force_bc_types = np.zeros(
            (self.nnodes, self.degrees_freedom), dtype=np.intc)
        force_bc_values = np.zeros(
            (self.nnodes, self.degrees_freedom), dtype=np.float64)
        tip_types = np.zeros(
            (self.nnodes, self.degrees_freedom), dtype=np.intc)
        num_force_bc_nodes = 0
        for i in range(self.nnodes):
            bnd = is_displacement_boundary(self.coords[i][:])
            forces_bnd = is_forces_boundary(self.coords[i][:])
            tip = is_tip(self.coords[i][:])
            is_force_node = 0
            for j in range(self.degrees_freedom):
                forces_bnd_j = forces_bnd[j]
                bnd_j = bnd[j]
                tip_j = tip[j]
                # Define boundary types and values
                if bnd_j is not None:
                    bc_types[i, j] = np.intc(1)
                    bc_values[i, j] = np.float64(bnd_j)
                # Define forces boundary types and values
                if forces_bnd_j is not None:
                    is_force_node = 1
                    force_bc_types[i, j] = np.intc(1)
                    force_bc_values[i, j] = np.float64(
                        forces_bnd_j / self.volume[i])
                # Define tip
                if tip_j is not None:
                    tip_types[i, j] = 1
            num_force_bc_nodes += is_force_node
        if num_force_bc_nodes != 0:
            force_bc_values = np.float64(
                np.divide(force_bc_values, num_force_bc_nodes))

        return (bc_types, bc_values, force_bc_types, force_bc_values,
                tip_types)

    def simulate(self, steps, u=None, ud=None, connectivity=None,
                 regimes=None, critical_stretch=None, bond_stiffness=None,
                 displacement_bc_magnitudes=None, force_bc_magnitudes=None,
                 max_load=0.0, first_step=1, write=None,
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
        :type regimes: :class:`numpy.ndarray`
        :arg critical_stretch: An (nregimes, nbond_types) array of critical
            stretch values, each corresponding to a bond type and a regime,
            or a float value of the critical stretch of the Peridynamic
            bond-based prototype microelastic brittle (PMB) model.
        :type critical_stretch: :class:`numpy.ndarray` or float
        :arg bond_stiffness: An (nregimes, nbond_types) array of bond
            stiffness values, each corresponding to a bond type and a regime,
            or a float value of the bond stiffness the Peridynamic bond-based
            prototype microelastic brittle (PMB) model.
        :type bond_stiffness: :class:`numpy.ndarray` or float
        :arg displacement_bc_magnitudes: (steps, ) array of the magnitude
            applied to the displacement boundary conditions over time.
        :type displacement_bc_magnitudes: :class:`numpy.ndarray`
        :arg force_bc_magnitudes: (steps, ) array of the magnitude applied to
            the force boundary conditions over time.
        :type force_bc_magnitudes: :class:`numpy.ndarray`
        :arg int first_step: The starting step number. This is useful when
            restarting a simulation.
        :arg int write: The frequency, in number of steps, to write the system
            to a mesh file by calling :meth:`Model.write_mesh`. If `None` then
            no output is written. Default `None`.
        :arg write_path: The path where the periodic mesh files should be
            written.
        :type write_path: path-like or str

        :returns: A tuple of the final displacements (`u`), damage,
            the final velocities (`ud`) connectivity, a (steps, ) list of the
            total sum of all damage over the time steps, a (steps, 3) array of
            the tip displacements over the time-steps and a (steps, 3) array of
            the tip resultant force over the time-steps.
            the current value of ease_off
        :rtype: tuple(
            :class:`numpy.ndarray`, :class:`numpy.ndarray`,
            :class:`numpy.ndarray`, tuple(:class:`numpy.ndarray`,
            :class:`numpy.ndarray`))
        """
        (u,
         ud,
         force,
         nlist,
         n_neigh,
         displacement_bc_magnitudes,
         force_bc_magnitudes,
         damage,
         damage_sum_data,
         tip_displacement_data,
         tip_velocity_data,
         tip_force_data,
         write_path) = self._simulate_initialise(
             steps, first_step, regimes, u, ud, displacement_bc_magnitudes,
             force_bc_magnitudes, connectivity, bond_stiffness,
             critical_stretch, write_path)

        for step in trange(first_step, first_step+steps,
                           desc="Simulation Progress", unit="steps"):

            # Call one integration step
            self.integrator(
                displacement_bc_magnitudes[step - 1],
                force_bc_magnitudes[step - 1])

            if write:
                if step % write == 0:
                    (u,
                     ud,
                     force,
                     damage,
                     nlist,
                     n_neigh) = self.integrator.write(
                         u, ud, force, damage, nlist, n_neigh)

                    self.write_mesh(write_path/f"U_{step}.vtk", damage, u)

                    tip_displacement = 0
                    tip_velocity = 0
                    tip_force = 0
                    tmp = 0
                    for i in range(self.nnodes):
                        for j in range(self.degrees_freedom):
                            if self.tip_types[i][j] == 1:
                                tmp += 1
                                tip_displacement += u[i][j]
                                tip_force += force[i][j]
                                tip_velocity += ud[i][j]
                    if tmp != 0:
                        tip_displacement /= tmp
                        tip_velocity /= tmp
                    else:
                        tip_force = None
                        tip_displacement = None
                        tip_velocity = None

                    tip_displacement_data.append(tip_displacement)
                    tip_velocity_data.append(tip_velocity)
                    tip_force_data.append(tip_force)
                    damage_sum = np.sum(damage)
                    damage_sum_data.append(damage_sum)
                    if damage_sum > 0.05*self.nnodes:
                        warnings.warn('Over 5% of bonds have broken!\
                                      peridynamics simulation continuing')
                    elif damage_sum > 0.7*self.nnodes:
                        warnings.warn('Over 7% of bonds have broken!\
                                      peridynamics simulation continuing')
        (u,
         ud,
         force,
         damage,
         nlist,
         n_neigh) = self.integrator.write(
             u, ud, force, damage, nlist, n_neigh)

        return (u, damage, (nlist, n_neigh), force, ud, damage_sum_data,
                tip_displacement_data, tip_velocity_data, tip_force_data)

    def _simulate_initialise(
            self, steps, first_step, regimes, u, ud,
            displacement_bc_magnitudes, force_bc_magnitudes, connectivity,
            bond_stiffness, critical_stretch, write_path):
        """
        Initialise simulation variables.

        :arg int steps: The number of simulation steps to conduct.
        :arg regimes: The initial regimes for the simulation. A
            (`nodes`, `max_neighbours`) array of type
            :class:`numpy.ndarray` of the regimes of the bonds
            of a neighbour list and the number of neighbours for each node.
        :type regimes: :class:`numpy.ndarray`
        :arg u: The initial displacements for the simulation. If `None` the
            displacements will be initialised to zero. Default `None`.
        :type u: :class:`numpy.ndarray`
        :arg ud: The initial velocities for the simulation. If `None` the
            velocities will be initialised to zero. Default `None`.
        :type ud: :class:`numpy.ndarray`
        :arg displacement_bc_magnitudes: (steps, ) array of the magnitude
            applied to the displacement boundary conditions over time.
        :type displacement_bc_magnitudes: :class:`numpy.ndarray`
        :arg force_bc_magnitudes: (steps, ) array of the magnitude applied to
            the force boundary conditions over time.
        :type force_bc_magnitudes: :class:`numpy.ndarray`
        :arg connectivity: The initial connectivity for the simulation. A tuple
            of a neighbour list and the number of neighbours for each node. If
            `None` the connectivity at the time of construction of the
            :class:`Model` object will be used. Default `None`.
        :type connectivity: tuple(:class:`numpy.ndarray`,
            :class:`numpy.ndarray`)
        :arg bond_stiffness: An (nregimes, nbond_types) array of bond
            stiffness values, each corresponding to a bond type and a regime.
        :type bond_stiffness: list or :class: `numpy.ndarray`
        :arg critical_stretch: An (nregimes, nbond_types) array of critical
            stretch values, each corresponding to a bond type and a regime.
        :type critical_stretch: list or :class: `numpy.ndarray`
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
                     :class:`numpy.ndarray`, :class`pathlib.Path`)
        """
        # Create initial displacements if none is provided
        if u is None:
            u = np.zeros((self.nnodes, 3), dtype=np.float64)
        if ud is None:
            ud = np.zeros((self.nnodes, 3), dtype=np.float64)
        # Initiate forces and damage
        force = np.zeros((self.nnodes, 3), dtype=np.float64)
        damage = np.zeros(self.nnodes, dtype=np.float64)
        # Create boundary condition magnitudes if none is provided
        if displacement_bc_magnitudes is None:
            displacement_bc_magnitudes = np.zeros(
                first_step + steps - 1, dtype=np.float64)
        elif type(displacement_bc_magnitudes) == np.ndarray:
            if len(displacement_bc_magnitudes) < steps:
                raise ValueError("displacement_bc_magnitudes length must be "
                                 "equal to or greater than (first_step + steps"
                                 " - 1), (expected {}, got {})".format(
                                     first_step + steps - 1,
                                     len(displacement_bc_magnitudes)))
            displacement_bc_magnitudes = displacement_bc_magnitudes.astype(
                np.float64)
        else:
            raise TypeError("displacement_bc_magnitudes type is wrong "
                            "(expected {}, got {})".format(
                                np.ndarray,
                                type(displacement_bc_magnitudes)))
        if force_bc_magnitudes is None:
            force_bc_magnitudes = np.zeros(
                first_step + steps - 1, dtype=np.float64)
        elif type(force_bc_magnitudes) == np.ndarray:
            if len(force_bc_magnitudes) < steps:
                raise ValueError("force_bc_magnitudes length must be "
                                 "equal to or greater than (first_step + steps"
                                 " - 1), (expected {}, got {})".format(
                                     first_step + steps - 1,
                                     len(force_bc_magnitudes)))
            force_bc_magnitudes = force_bc_magnitudes.astype(
                np.float64)
        else:
            raise TypeError("force_bc_magnitudes type is wrong "
                            "(expected {}, got {})".format(
                                np.ndarray,
                                type(force_bc_magnitudes)))
        # Use the initial connectivity (when the Model was constructed) if none
        # is provided
        if connectivity is None:
            nlist, n_neigh = self.initial_connectivity
        elif type(connectivity) == tuple:
            if len(connectivity) != 2:
                raise ValueError("connectivity size is wrong (expected 2,"
                                 " got {})".format(len(connectivity)))
            nlist, n_neigh = connectivity
        else:
            raise TypeError("connectivity type is wrong (expected {} or"
                            " {}, got {})".format(
                                    tuple, type(None), type(connectivity)))
        # Use the initial regimes of linear elastic (0 values) if none
        # is provided
        if regimes is None:
            regimes = np.zeros(
                (self.nnodes, self.max_neighbours), dtype=np.intc)
        elif type(regimes) == np.ndarray:
            if np.shape(regimes) != (self.nnodes, self.max_neighbours):
                raise ValueError("regimes shape is wrong, and must be "
                                 "(nnodes, max_neighbours) "
                                 "(expected {}, got {})".format(
                                     (self.nnodes, self.max_neighbours),
                                     np.shape(regimes)))
            regimes = regimes.astype(np.intc)
        else:
            raise TypeError("regimes type is wrong "
                            "(expected {} or {}, got {})".format(
                                np.ndarray,
                                type(None),
                                type(regimes)))

        # Use the initial damage model
        # (when the Model was constructed) if none is provided
        if ((bond_stiffness is None) and (critical_stretch is None)):
            bond_stiffness = self.bond_stiffness
            critical_stretch = self.critical_stretch
            plus_cs = self.plus_cs
            nbond_types = self.nbond_types
            nregimes = self.nregimes
        else:
            print('here')
            (bond_stiffness,
             critical_stretch,
             plus_cs,
             nbond_types,
             nregimes) = self._set_damage_model(
                 bond_stiffness, critical_stretch)
            if nbond_types != self.nbond_types:
                raise ValueError(
                    "Number of bond types has unexpectedly changed from when "
                    " the model was constructed. Please reinstantiate "
                    ":class:`Model` with the new number of bond types "
                    "(expected {}, got {}).".format(
                        self.nbond_types, nbond_types))

        # If no write path was provided use the current directory, otherwise
        # ensure write_path is a Path object.
        if write_path is None:
            write_path = pathlib.Path()
        else:
            write_path = pathlib.Path(write_path)

        # Container for plotting data
        damage_sum_data = []
        tip_displacement_data = []
        tip_velocity_data = []
        tip_force_data = []

        # Initialise the OpenCL buffers
        self.integrator.set_buffers(
            nlist, n_neigh, bond_stiffness, critical_stretch, plus_cs, u, ud,
            force, damage, regimes, nregimes, nbond_types)

        return (u, ud, force, nlist, n_neigh, displacement_bc_magnitudes,
                force_bc_magnitudes, damage,
                damage_sum_data, tip_displacement_data, tip_velocity_data,
                tip_force_data, write_path)


def initial_crack_helper(crack_function):
    """
    Help the construction of an initial crack function.

    `crack_function` has the form `crack_function(icoord, jcoord)` where
    `icoord` and `jcoord` are :class:`numpy.ndarray` s representing two node
    coordinates.  crack_function returns a truthy value if there is a crack
    between the two nodes and a falsy value otherwise.

    This decorator returns a function which takes all node coordinates and
    returns a list of tuples of the indices pair of nodes which define the
    crack. This function can therefore be used as the `initial_crack` argument
    of the :class:`Model`

    :arg function crack_function: The function which determine whether there is
        a crack between a pair of node coordinates.

    :returns: A function which determines all pairs of nodes with a crack
        between them.
    :rtype: function
    """
    def initial_crack(coords, nlist, n_neigh):
        crack = []

        # Get all pairs of bonded particles
        nnodes = nlist.shape[0]
        pairs = [(i, j) for i in range(nnodes) for j in nlist[i][0:n_neigh[i]]
                 if i < j]

        # Check each pair using the crack function
        for i, j in pairs:
            if crack_function(coords[i], coords[j]):
                crack.append((i, j))
        return crack
    return initial_crack


class DimensionalityError(Exception):
    """An invalid dimensionality argument used to construct a model."""

    def __init__(self, dimensions):
        """
        Construct the exception.

        :arg int dimensions: The number of dimensions passed as an argument to
            :meth:`Model`.

        :rtype: :class:`DimensionalityError`
        """
        message = (
                "The number of dimensions must be 2 or 3,"
                f" {dimensions} was given."
                )

        super().__init__(message)


class FamilyError(Exception):
    """One or more nodes have no bonds in the initial state."""

    def __init__(self, family):
        """
        Construct the exception.

        :arg family: The family array.
        :type family: :class:`numpy.ndarray`

        :rtype: :class:`FamilyError`
        """
        indicies = np.where(family == 0)[0]
        indicies = " ".join([f"{index}" for index in indicies])
        message = (
                "The following nodes have no bonds in the initial state,"
                f" {indicies}."
                )

        super().__init__(message)


class InvalidIntegrator(Exception):
    """An invalid integrator has been passed to `simulate`."""

    def __init__(self, integrator):
        """
        Construct the exception.

        :arg integrator: The object passed to :meth:`Model.simulate` as the
            integrator argument.

        :rtype: :class:`InvalidIntegrator`
        """
        message = (
                f"{integrator} is not an instance of"
                "peridynamics.integrators.Integrator"
                )

        super().__init__(message)

"""Peridynamics model."""
from .integrators import Integrator
from .neighbour_list import (family, create_neighbour_list_BB, break_bonds,
                             create_crack)
from .peridynamics import damage, bond_force
from collections import namedtuple
import meshio
import numpy as np
import pathlib
from tqdm import trange


_MeshElements = namedtuple("MeshElements", ["connectivity", "boundary"])
_mesh_elements_2d = _MeshElements(connectivity="triangle",
                                  boundary="line")
_mesh_elements_3d = _MeshElements(connectivity="tetra",
                                  boundary="triangle")


class Model(object):
    """
    A peridynamics model.

    This class allows users to define a peridynamics system from parameters and
    a set of initial conditions (coordinates and connectivity).

        >>> from peridynamics import Model
        >>>
        >>> model = Model(
        >>>     mesh_file="./example.msh",
        >>>     horizon=0.1,
        >>>     critical_strain=0.005,
        >>>     elastic_modulus=0.05
        >>>     )

    To define a crack in the inital configuration, you may supply a list of
    pairs of particles between which the crack is.

        >>> initial_crack = [(1,2), (5,7), (3,9)]
        >>> model = Model(
        >>>     mesh_file="./example.msh",
        >>>     horizon=0.1,
        >>>     critical_strain=0.005,
        >>>     elastic_modulus=0.05,
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
        >>>     mesh_file="./example.msh",
        >>>     horizon=0.1,
        >>>     critical_strain=0.005,
        >>>     elastic_modulus=0.05,
        >>>     initial_crack=initial_crack
        >>>     )

    The :meth:`Model.simulate` method can be used to conduct a peridynamics
    simulation. For this an :class:`peridynamics.integrators.Integrator` is
    required, and optionally a function implementing the boundary conditions.

        >>> from peridynamics.integrators import Euler
        >>>
        >>> model = Model(...)
        >>>
        >>> euler = Euler(dt=1e-3)
        >>>
        >>> indices = np.arange(model.nnodes)
        >>> model.lhs = indices[
        >>>     model.coords[:, 0] < 1.5*model.horizon
        >>>     ]
        >>> model.rhs = indices[
        >>>     model.coords[:, 0] > 1.0 - 1.5*model.horizon
        >>>     ]
        >>>
        >>> def boundary_function(model, u, step):
        >>>     u[model.lhs] = 0
        >>>     u[model.rhs] = 0
        >>>     u[model.lhs, 0] = -1.0 * step
        >>>     u[model.rhs, 0] = 1.0 * step
        >>>
        >>>     return u
        >>>
        >>> u, damage, *_ = model.simulate(
        >>>     steps=1000,
        >>>     integrator=euler,
        >>>     boundary_function=boundary_function
        >>>     )
    """

    def __init__(self, mesh_file, horizon, critical_strain, elastic_modulus,
                 transfinite = 0, volume_total=None,
                 connectivity=None, initial_crack=[], dimensions=2):
        """
        Construct a :class:`Model` object.

        :arg str mesh_file: Path of the mesh file defining the systems nodes
            and connectivity.
        :arg float horizon: The horizon radius. Nodes within `horizon` of
            another interact with that node and are said to be within its
            neighbourhood.
        :arg float critical_strain: The critical strain of the model. Bonds
            which exceed this strain are permanently broken.
        :arg float elastic_modulus: The appropriate elastic modulus of the
            material.
        :arg bool transfinite: Is the mesh a transfinite(=1) or a triangular/
            tetra(=0). Default 0. Tranfinite mode approximates the volumes of 
            the nodes as the average volume of nodes on a cuboidal tensor-grid 
            mesh.
        :arg float volume_total: Total volume of the mesh. Must be provided if 
            transfinite mode (transfinite=1) is used.
        :arg initial_crack: The initial crack of the system. The argument may
            be a list of tuples where each tuple is a pair of integers
            representing nodes between which to create a crack. Alternatively,
            the arugment may be a function which takes the (nnodes, 3)
            :class:`numpy.ndarray` of coordinates as an argument, and returns a
            list of tuples defining the initial crack. Default is []
        :type initial_crack: list(tuple(int, int)) or function
        :arg int dimensions: The dimensionality of the model. The
            default is 2.

        :returns: A new :class:`Model` object.
        :rtype: Model

        :raises DimensionalityError: when an invalid `dimensions` argument is
            provided.
        :raises FamilyError: when a node has no neighbours (other nodes it
            interacts with) in the initial state.
        """
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
        self.critical_strain = critical_strain

        # Determine bond stiffness
        self.bond_stiffness = (
            18.0 * elastic_modulus / (np.pi * self.horizon**4)
            )

        if transfinite:
            if volume_total is None:
                raise ValueError("If the mesh is regular cuboidal tensor grid\
                                 (transfinite), a total volume (key word arg\
                                'volume_total') must be provided")
        # Calculate the volume for each node
        self.volume, self.sum_total_volume = self._volume(
            transfinite, volume_total)

        # Calculate the family (number of bonds in the initial configuration)
        # for each node
        self.family = family(self.coords, horizon)
        if np.any(self.family == 0):
            raise FamilyError(self.family)

        if connectivity is None:
            # Create the neighbourlist
            # Maximum number of nodes that any one of the nodes is connected
            # to, must be a power of 2 (for OpenCL reduction)
            self.max_neighbours = np.intc(
                        1<<(self.family.max()-1).bit_length()
                    )
            connectivity = create_neighbour_list_BB(
                self.coords, horizon, self.max_neighbours
                )
        elif type(connectivity) == tuple:
            if len(connectivity) != 2:
                raise ValueError("connectivity must be of size 2")
            nlist, n_neigh = connectivity
        else:
            raise TypeError("connectivity must be a tuple or None")

        # Initialise initial crack
        if initial_crack:
            if callable(initial_crack):
                initial_crack = initial_crack(self.coords, nlist, n_neigh)
            create_crack(
                np.array(initial_crack, dtype=np.int32),  nlist, n_neigh
                )
        self.initial_connectivity = (nlist, n_neigh)

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
        
        :arg float volume_total: User input for the total volume of the mesh, for checking
        sum total of elemental volumes is equal to user input volume for simple
        prismatic problems. In the case where no expected total volume is provided,
        the check is not done.
        :arg bool transfinite: Is the mesh a transfinite(=1) or a triangular/tetra(=0).

        :returns: None
        :rtype: NoneType
        """
        volume = np.zeros(self.nnodes)
        dimensions = self.dimensions
        sum_total_volume = 0.0

        if transfinite:
            tmp = volume_total / self.nnodes
            volume = tmp * np.ones(self.nnodes)
            sum_total_volume = volume_total
        elif dimensions == 2:
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
        
        return volume, sum_total_volume

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
        break_bonds(self.coords+u, self.coords, nlist, n_neigh,
                    self.critical_strain)

    def _damage(self, n_neigh):
        """
        Calculate bond damage.

        :arg n_neigh: The number of neighbours of each node.
        :type n_neigh: :class:`numpy.ndarray`

        :returns: A (`nnodes`, ) array containing the damage for each node.
        :rtype: :class:`numpy.ndarray`
        """
        return damage(n_neigh, self.family)

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
        f = bond_force(self.coords+u, self.coords, nlist, n_neigh,
                       self.volume, self.bond_stiffness)

        return f

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
        (nlist,
         n_neigh,
         u,
         boundary_function,
         write_path) = self._simulate_initialise(
            integrator, boundary_function, u, connectivity, write_path
            )

        for step in trange(first_step, first_step+steps,
                           desc="Simulation Progress", unit="steps"):

            # Calculate the force due to bonds on each node
            force = self._bond_force(u, nlist, n_neigh)

            # Conduct one integration step
            u = integrator(u, force)
            # Apply boundary conditions
            u = boundary_function(self, u, step)

            # Update neighbour list
            self._break_bonds(u, nlist, n_neigh)

            # Calculate the current damage
            damage = self._damage(n_neigh)

            if write:
                if step % write == 0:
                    self.write_mesh(write_path/f"U_{step}.vtk", damage, u)

        return u, damage, (nlist, n_neigh)

    def _simulate_initialise(self, integrator, boundary_function, u,
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

        return nlist, n_neigh, u, boundary_function, write_path


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

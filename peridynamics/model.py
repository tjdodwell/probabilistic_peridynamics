"""Peridynamics model."""
from .integrators import Integrator
from collections import namedtuple
import meshio
import numpy as np
import pathlib
from scipy import sparse
from scipy.spatial.distance import cdist
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

    :Example: ::

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

    :Example: ::

        >>> from peridynamics import Model, initial_crack_helper
        >>>
        >>> initial_crack = [(1,2), (5,7), (3,9)]
        >>> model = Model(mesh_file, horizon=0.1, critical_strain=0.005,
        >>>               elastic_modulus=0.05, initial_crack=initial_crack)

    If it is more convenient to define the crack as a function you may also
    pass a function to the constructor which takes the array of coordinates as
    its only argument and returns a list of tuples as described above. The
    :func:`peridynamics.model.initial_crack_helper` decorator has been provided
    to easily create a function of the correct form from one which tests a
    single pair of node coordinates and returns `True` or `False`.

    :Example: ::

        >>> from peridynamics import Model, initial_crack_helper
        >>>
        >>> @initial_crack_helper
        >>> def initial_crack(x, y):
        >>>     ...
        >>>     if crack:
        >>>         return True
        >>>     else:
        >>>         return False
        >>>
        >>> model = Model(mesh_file, horizon=0.1, critical_strain=0.005,
        >>>               elastic_modulus=0.05, initial_crack=initial_crack)

    The :meth:`Model.simulate` method can be used to conduct a peridynamics
    simulation. For this an :class:`peridynamics.integrators.Integrator` is
    required, and optionally a function implementing the boundary conditions.

    :Example: ::

        >>> from peridynamics import Model, initial_crack_helper
        >>> from peridynamics.integrators import Euler
        >>>
        >>> model = Model(...)
        >>>
        >>> euler = Euler(dt=1e-3)
        >>>
        >>> indices = np.arange(model.nnodes)
        >>> model.lhs = indices[model.coords[:, 0] < 1.5*model.horizon]
        >>> model.rhs = indices[model.coords[:, 0] > 1.0 - 1.5*model.horizon]
        >>>
        >>> def boundary_function(model, u, step):
        >>>     u[model.lhs] = 0
        >>>     u[model.rhs] = 0
        >>>     u[model.lhs, 0] = -1.0 * step
        >>>     u[model.rhs, 0] = 1.0 * step
        >>>
        >>>     return u
        >>>
        >>> u, damage, *_ = model.simulate(steps=1000, integrator=euler,
        >>>                                boundary_function=boundary_function)
    """

    def __init__(self, mesh_file, horizon, critical_strain, elastic_modulus,
                 initial_crack=[], dimensions=2):
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

        # Calculate the volume for each node
        self.volume = self._volume()

        # Determine neighbours
        neighbourhood = self._neighbourhood()

        # Set family, the number of neighbours for each node
        self.family = np.squeeze(np.array(np.sum(neighbourhood, axis=0)))
        if np.any(self.family == 0):
            raise FamilyError(self.family)

        # Set the initial connectivity
        self.initial_connectivity = self._connectivity(neighbourhood,
                                                       initial_crack)

        # Set the node distance and failure strain matrices
        _, _, _, self.L_0 = self._H_and_L(self.coords,
                                          self.initial_connectivity)

    def _read_mesh(self, filename):
        """
        Read the model's nodes, connectivity and boundary from a mesh file.

        :arg str filename: Path of the mesh file to read

        :returns: None
        :rtype: NoneType
        """
        mesh = meshio.read(filename)

        # Get coordinates, encoded as mesh points
        self.coords = mesh.points
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

    def _volume(self):
        """
        Calculate the value of each node.

        :returns: None
        :rtype: NoneType
        """
        volume = np.zeros(self.nnodes)
        dimensions = self.dimensions

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
            elif dimensions == 3:
                a, b, c, d = self.coords[nodes]

                # Volume of a tetrahedron
                i = a - d
                j = b - d
                k = c - d
                element_volume = abs(np.dot(i, np.cross(j, k))) / 6

            # Add fraction element volume to all nodes belonging to that
            # element
            volume[nodes] += element_volume / element_nodes

        return volume

    def _neighbourhood(self):
        """
        Determine the neighbourhood of all nodes.

        :returns: The sparse neighbourhood matrix.  Element [i, j] of this
            martrix is True if i is within `horizon` of j and False otherwise.
        :rtype: :class:`scipy.sparse.csr_matrix`
        """
        # Calculate the Euclidean distance between each pair of nodes
        distance = cdist(self.coords, self.coords, 'euclidean')

        # Construct the neighbourhood matrix (neighbourhood[i, j] = True if i
        # and j are neighbours)
        nnodes = self.nnodes
        neighbourhood = np.zeros((nnodes, nnodes), dtype=np.bool)
        # Connect nodes which are within horizon of each other
        neighbourhood[distance < self.horizon] = True
        # Remove self interaction
        np.fill_diagonal(neighbourhood, False)

        return sparse.csr_matrix(neighbourhood)

    def _connectivity(self, neighbourhood, initial_crack):
        """
        Initialise the connectivity.

        :arg neighbourhood: The sparse neighbourhood matrix.
        :type neighbourhood: :class:`scipy.sparse.csr_matrix`
        :arg initial_crack: The initial crack of the system. The argument may
            be a list of tuples where each tuple is a pair of integers
            representing nodes between which to create a crack. Alternatively,
            the arugment may be a function which takes the (nnodes, 3)
            :class:`numpy.ndarray` of coordinates as an argument, and returns a
            list of tuples defining the initial crack.
        :type initial_crack: list(tuple(int, int)) or function

        :returns: The sparse connectivity matrix. Element [i, j] of this matrix
            is True if i and j are bonded and False otherwise.
        :rtype: :class:`scipy.sparse.csr_matrix`
        """
        if callable(initial_crack):
            initial_crack = initial_crack(self.coords, neighbourhood)

        # Construct the initial connectivity matrix
        conn = neighbourhood.toarray()
        for i, j in initial_crack:
            # Connectivity is symmetric
            conn[i, j] = False
            conn[j, i] = False

        # Lower triangular - count bonds only once
        conn = np.tril(conn, -1)

        # Convert to sparse matrix
        return sparse.csr_matrix(conn)

    @staticmethod
    def _displacements(r):
        """
        Determine displacements, in each dimension, between vectors.

        :arg r: A (n,3) array of coordinates.
        :type r: :class:`numpy.ndarray`

        :returns: A tuple of three arrays giving the displacements between
            each pair of parities in the first, second and third dimensions
            respectively. m[i, j] is the distance from j to i (i.e. i - j).
        :rtype: tuple(:class:`numpy.ndarray`)
        """
        n = len(r)
        x = np.tile(r[:, 0], (n, 1))
        y = np.tile(r[:, 1], (n, 1))
        z = np.tile(r[:, 2], (n, 1))

        d_x = x.T - x
        d_y = y.T - y
        d_z = z.T - z

        return d_x, d_y, d_z

    def _H_and_L(self, r, connectivity):
        """
        Construct the displacement and distance matrices.

        The H matrices  are sparse matrices containing displacements in
        a particular dimension and the L matrix isa sparse matrix containing
        the Euclidean distance. Elements for particles which are not connected
        are 0.

        :arg r: The positions of all nodes.
        :type r: :class:`numpy.ndarray`
        :arg connectivity: The sparse connectivity matrix.
        :type connectivity: :class:`scipy.sparse.csr_matrix`

        :returns: (H_x, H_y, H_z, L) A tuple of sparse matrix. H_x, H_y and H_z
            are the matrices of displacements between pairs of particles in the
            x, y and z dimensions respectively. L is the Euclidean distance
            between pairs of particles.
        :rtype: tuple(:class:`scipy.sparse.csr_matrix`)
        """
        # Get displacements in each dimension between coordinate
        H_x, H_y, H_z = self._displacements(r)

        # Convert to spare matrices filtered by the connectivity matrix (i.e.
        # only for particles which interact).
        a = connectivity + connectivity.transpose()
        H_x = a.multiply(H_x)
        H_y = a.multiply(H_y)
        H_z = a.multiply(H_z)

        L = (H_x.power(2) + H_y.power(2) + H_z.power(2)).sqrt()

        return H_x, H_y, H_z, L

    def _strain(self, L):
        """
        Calculate the strain of all bonds for a given displacement.

        :arg L: The euclidean distance between each pair of nodes.
        :type L: :class:`scipy.sparse.csr_matrix`

        :returns: The strain between each pair of nodes.
        :rtype: :class:`scipy.sparse.lil_matrix`
        """
        # Calculate difference in bond lengths from the initial state
        dL = L - self.L_0

        # Calculate strain
        nnodes = self.nnodes
        strain = sparse.lil_matrix((nnodes, nnodes))
        non_zero = self.L_0.nonzero()
        strain[non_zero] = dL[non_zero]/self.L_0[non_zero]

        return strain

    def _break_bonds(self, strain, connectivity):
        """
        Break bonds which have exceeded the critical strain.

        :arg strain: The strain of each bond.
        :type strain: :class:`scipy.sparse.lil_matrix`
        :arg connectivity: The sparse connectivity matrix.
        :type connectivity: :class:`scipy.sparse.csr_matrix`

        :returns: The updated connectivity.
        :rtype: :class:`scipy.sparse.csr_matrix`
        """
        unbroken = sparse.lil_matrix(connectivity.shape)

        # Find broken bonds
        nnodes = self.nnodes
        critical_strains = np.full((nnodes, nnodes), self.critical_strain)
        connected = connectivity.nonzero()
        unbroken[connected] = (
                critical_strains[connected] - abs(strain[connected])
                ) > 0

        connectivity = sparse.csr_matrix(unbroken)

        return connectivity

    def _damage(self, connectivity):
        """
        Calculate bond damage.

        :arg connectivity: The sparse connectivity matrix.
        :type connectivity: :class:`scipy.sparse.csr_matrix`

        :returns: A (`nnodes`, ) array containing the damage for each node.
        :rtype: :class:`numpy.ndarray`
        """
        family = self.family
        # Sum all unbroken bonds for each node
        unbroken_bonds = (connectivity + connectivity.transpose()).sum(axis=0)
        # Convert matrix object to array
        unbroken_bonds = np.squeeze(np.array(unbroken_bonds))

        # Calculate damage for each node
        damage = np.divide((family - unbroken_bonds), family)

        return damage

    def _bond_force(self, strain, connectivity, L, H_x, H_y, H_z):
        """
        Calculate the force due to bonds acting on each node.

        :arg strain: The strain of each bond.
        :type strain: :class:`scipy.sparse.lil_matrix`
        :arg connectivity: The sparse connectivity matrix.
        :type connectivity: :class:`scipy.sparse.csr_matrix`
        :arg L: The Euclidean distance between pairs of nodes.
        :type L: :class:`scipy.sparse.csr_matrix`
        :arg H_x: The displacement in the x dimension between each pair of
            nodes.
        :type H_x: :class:`scipy.sparse.csr_matrix`
        :arg H_y: The displacement in the y dimension between each pair of
            nodes.
        :type H_y: :class:`scipy.sparse.csr_matrix`
        :arg H_z: The displacement in the z dimension between each pair of
            nodes.
        :type H_z: :class:`scipy.sparse.csr_matrix`

        :returns: A (`nnodes`, 3) array of the component of the force in each
            dimension for each node.
        :rtype: :class:`numpy.ndarray`
        """
        # Calculate the normalised forces
        force_normd = sparse.lil_matrix(connectivity.shape)
        connected = connectivity.nonzero()
        force_normd[connected] = strain[connected] / L[connected]

        # Make lower triangular into full matrix
        force_normd.tocsr()
        force_normd = force_normd + force_normd.transpose()

        # Calculate component of force in each dimension
        bond_force_x = force_normd.multiply(H_x)
        bond_force_y = force_normd.multiply(H_y)
        bond_force_z = force_normd.multiply(H_z)

        # Calculate total force on nodes in each dimension
        F_x = np.squeeze(np.array(bond_force_x.sum(axis=0)))
        F_y = np.squeeze(np.array(bond_force_y.sum(axis=0)))
        F_z = np.squeeze(np.array(bond_force_z.sum(axis=0)))

        # Determine actual force
        F = np.stack((F_x, F_y, F_z), axis=-1)
        F *= self.volume.reshape((self.nnodes, 1))
        F *= self.bond_stiffness

        return F

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
        :arg connectivity: The initial connectivity for the simulation. If
            `None` the connectivity at the time of construction of the
            :class:`Model` object will be used.
        :type connectivity: :class:`scipy.sparse.csr_matrix` or
            :class:`numpy.ndarray`
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
            :class:`scipy.sparse.csr_matrix`)
        """
        if not isinstance(integrator, Integrator):
            raise InvalidIntegrator(integrator)

        # Create initial displacements is none is provided
        if u is None:
            u = np.zeros((self.nnodes, 3))

        # Use the initial connectivity (when the Model was constructed) if none
        # is provided
        if connectivity is None:
            connectivity = self.initial_connectivity
        elif type(connectivity) == np.ndarray:
            connectivity = sparse.csr_matrix(connectivity)

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

        for step in trange(first_step, first_step+steps,
                           desc="Simulation Progress", unit="steps"):
            # Get current distance between nodes (i.e. accounting for
            # displacements)
            H_x, H_y, H_z, L = self._H_and_L(self.coords+u, connectivity)

            # Calculate the strain of each bond
            strain = self._strain(L)

            # Update the connectivity and calculate the current damage
            connectivity = self._break_bonds(strain, connectivity)
            damage = self._damage(connectivity)

            # Calculate the bond due to forces on each node
            f = self._bond_force(strain, connectivity, L, H_x, H_y, H_z)

            # Conduct one integration step
            u = integrator(u, f)

            # Apply boundary conditions
            u = boundary_function(self, u, step)

            if write:
                if step % write == 0:
                    self.write_mesh(write_path/f"U_{step}.vtk", damage, u)

        return u, damage, connectivity


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
    def initial_crack(coords, neighbourhood):
        crack = []
        # Get all pairs of bonded particles (within horizon) where i < j and
        # i /= j (using the upper triangular portion)
        i, j = sparse.triu(neighbourhood, 1).nonzero()
        # Check each pair using the crack function
        for i, j in zip(i, j):
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
                f"The number of dimensions must be 2 or 3,"
                " {dimensions} was given."
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

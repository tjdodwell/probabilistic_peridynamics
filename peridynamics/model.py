from .integrators import Integrator
from collections import namedtuple
from itertools import combinations
import meshio
import numpy as np
from scipy import sparse
from scipy.spatial.distance import cdist


_MeshElements = namedtuple("MeshElements", ["connectivity", "boundary"])
_mesh_elements_2d = _MeshElements(connectivity="triangle",
                                  boundary="line")
_mesh_elements_3d = _MeshElements(connectivity="tetrahedron",
                                  boundary="triangle")


class Model:
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

    The :meth:`peridynamics.model.Model.simulate` method can be used to conduct
    a peridynamics simulation. For this an
    :class:`peridynamics.integrators.Integrator` is required, and optionally a
    function implementing the boundary conditions.

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
        >>> u, damage = model.simulate(steps=1000, integrator=euler,
        >>>                            boundary_function=boundary_function)
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
        self.neighbourhood = self._neighbourhood()

        # Set family, the number of neighbours for each node
        self.family = np.sum(self.neighbourhood, axis=0)

        # Set the connectivity
        self.connectivity = self._connectivity(self.neighbourhood,
                                               initial_crack)

        # Set the node distance and failure strain matrices
        _, _, _, self.L_0 = self._H_and_L(self.coords)

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
        Optionally, write damage and displacements as points data.

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

        for element in self.mesh_connectivity:
            # Compute area / volume
            val = 1. / len(element)

            # Define area of element
            if (self.dimensions == 2):
                xi, yi, *_ = self.coords[element[0]]
                xj, yj, *_ = self.coords[element[1]]
                xk, yk, *_ = self.coords[element[2]]
                val *= 0.5 * ((xj - xi) * (yk - yi) - (xk - xi) * (yj - yi))

            volume[element] += val

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

        return sparse.csr_matrix(neighbourhood)

    def _connectivity(self, neighbourhood, initial_crack):
        """
        Initialises the connectivity.

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
            initial_crack = initial_crack(self.coords)

        # Construct the initial connectivity matrix
        conn = neighbourhood.toarray()
        for i, j in initial_crack:
            # Connectivity is symmetric
            conn[i, j] = False
            conn[j, i] = False
        # Nodes are not connected with themselves
        np.fill_diagonal(conn, False)

        # Lower triangular - count bonds only once
        # make diagonal values False
        conn = np.tril(conn, -1)

        # Convert to sparse matrix
        return sparse.csr_matrix(conn)

    @staticmethod
    def _displacements(r):
        """
        Dertmine the displacment, in each dimension, between each pair of
        coordinates.

        :arg r: A (n,3) array of coordinates.
        :type r: :class:`numpy.ndarray`

        :returns: A tuple of three arrays giving the displacements between
            each pair of paritlces in the first, second and third dimensions
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

    def _H_and_L(self, r):
        """
        Constructs the H matrices (sparse matrices containing
        displacements in a particular dimension) and the L matrix (a sparse
        matrix containing the Euclidean distance). Elements for particles which
        are not connected are 0.

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
        a = self.connectivity
        a = a + a.transpose()
        H_x = sparse.csr_matrix(a.multiply(H_x))
        H_y = sparse.csr_matrix(a.multiply(H_y))
        H_z = sparse.csr_matrix(a.multiply(H_z))

        L = (H_x.power(2) + H_y.power(2) + H_z.power(2)).sqrt()

        return H_x, H_y, H_z, L

    def _strain(self, u, L):
        """
        Calculates the strain (bond stretch) of all nodes for a given
        displacement.

        :arg u: The displacement array with shape (`nnodes`, `dimension`).
        :type u: :class:`numpy.ndarray`
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
        strain[non_zero] = (dL[non_zero]/self.L_0[non_zero])

        return strain

    def damage(self, strain):
        """
        Calculates bond damage.

        :arg strain: The strain of each bond.
        :type strain: :class:`scipy.sparse.lil_matrix`

        :returns: A (`nnodes`, ) array containing the damage
            for each node.
        :rtype: :class:`numpy.ndarray`
        """
        connectivity = self.connectivity

        bond_healths = sparse.lil_matrix(connectivity.shape)

        # Find broken bonds
        nnodes = self.nnodes
        critical_strains = np.full((nnodes, nnodes), self.critical_strain)
        connected = connectivity.nonzero()
        bond_healths[connected] = (
            critical_strains[connected] - abs(strain[connected])
            ) > 0

        connectivity = sparse.csr_matrix(bond_healths)

        family = self.family
        # Sum all unbroken bonds for each node
        unbroken_bonds = (connectivity + connectivity.transpose()).sum(axis=0)
        # Convert matrix object to array
        unbroken_bonds = np.squeeze(np.array(unbroken_bonds))

        # Calculate damage for each node
        damage = np.divide((family - unbroken_bonds), family)

        return damage

    def bond_force(self, strain, L, H_x, H_y, H_z):
        """
        Calculate the force due to bonds acting on each node.

        :returns: A (`nnodes`, 3) array of the component of the force in each
            dimension for each node.
        :rtype: :class:`numpy.ndarray`
        """
        connectivity = self.connectivity

        # Step 1. Initiate container as a sparse matrix, only need calculate
        # for bonds that exist
        force_normd = sparse.lil_matrix(connectivity.shape)

        # Step 2. find normalised forces
        force_normd[connectivity.nonzero()] = (
            strain[connectivity.nonzero()]
            / L[connectivity.nonzero()]
            )

        # Make lower triangular into full matrix
        force_normd.tocsr()
        force_normd = force_normd + force_normd.transpose()

        # Multiply by the direction and scale of each bond (just trigonometry,
        # we have already scaled for bond length in step 2)
        bond_force_x = force_normd.multiply(H_x)
        bond_force_y = force_normd.multiply(H_y)
        bond_force_z = force_normd.multiply(H_z)

        # now sum along the rows to calculate resultant force on nodes
        F_x = np.array(bond_force_x.sum(axis=0))
        F_y = np.array(bond_force_y.sum(axis=0))
        F_z = np.array(bond_force_z.sum(axis=0))

        F_x.resize(self.nnodes)
        F_y.resize(self.nnodes)
        F_z.resize(self.nnodes)

        # Finally multiply by volume and stiffness
        F_x = self.bond_stiffness * np.multiply(F_x, self.volume)
        F_y = self.bond_stiffness * np.multiply(F_y, self.volume)
        F_z = self.bond_stiffness * np.multiply(F_z, self.volume)

        return np.stack((F_x, F_y, F_z), axis=-1)

    def simulate(self, steps, integrator, boundary_function=None, u=None,
                 write=None):
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

        if not isinstance(integrator, Integrator):
            raise InvalidIntegrator(integrator)

        # Create initial displacements is none is provided
        if u is None:
            u = np.zeros((self.nnodes, 3))

        # Create dummy boundary conditions function is none is provied
        if boundary_function is None:
            def boundary_function(model):
                return model.u

        for step in range(1, steps+1):
            # Get current distance between nodes (i.e. accounting for
            # displacements)
            H_x, H_y, H_z, L = self._H_and_L(self.coords+u)

            # Calculate bond stretch, damage and forces on nodes
            strain = self._strain(u, L)
            damage = self.damage(strain)
            f = self.bond_force(strain, L, H_x, H_y, H_z)

            # Conduct one integration step
            u = integrator(u, f)

            # Apply boundary conditions
            u = boundary_function(self, u, step)

            if write:
                if step % write == 0:
                    self.write_mesh(f"U_{step}.vtk", damage, u)

        return u, damage


def initial_crack_helper(crack_function):
    """
    A decorator to help with the construction of an initial crack function.

    crack_function has the form crack_function(icoord, jcoord) where icoord and
    jcoord are :class:`numpy.ndarray` s representing two node coordinates.
    crack_function returns a truthy value if there is a crack between the two
    nodes and a falsy value otherwise.

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
    def initial_crack(coords):
        crack = []
        # Iterate over all unique pairs of coordinates with their indicies
        for (i, icoord), (j, jcoord) in combinations(enumerate(coords), 2):
            if crack_function(icoord, jcoord):
                crack.append((i, j))
        return crack
    return initial_crack


class DimensionalityError(Exception):
    """
    Raised when an invalid dimensionality argument used to construct a model
    """
    def __init__(self, dimensions):
        message = (
            f"The number of dimensions must be 2 or 3,"
            " {dimensions} was given."
            )

        super().__init__(message)


class InvalidIntegrator(Exception):
    """
    Raised when the integrator passed to
    :meth:`peridynamics.model.Model.simulate` is not an instance of
    :class:`peridynamics.integrators.Integrator`.
    """
    def __init__(self, integrator):
        message = (
            f"{integrator} is not an instance of"
            "peridynamics.integrators.Integrator"
            )

        super().__init__(message)

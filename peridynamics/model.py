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

        self.bond_stiffness = (
            18.0 * elastic_modulus / (np.pi * self.horizon**4)
            )

        # Calculate the volume for each node
        self._set_volume()

        # Set the connectivity
        self._set_connectivity(initial_crack)

        # Set the node distance and failure strain matrices
        self._set_H()

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
        self.mesh_connectivity = mesh.cells[self.mesh_elements.connectivity]

        # Get boundary connectivity, mesh lines
        self.mesh_boundary = mesh.cells[self.mesh_elements.boundary]

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
            cells={
                self.mesh_elements.connectivity: self.mesh_connectivity,
                self.mesh_elements.boundary: self.mesh_boundary
                },
            point_data={
                "damage": damage,
                "displacements": displacements
                },
            file_format=file_format
            )

    def _set_volume(self):
        """
        Calculate the value of each node.

        :returns: None
        :rtype: NoneType
        """
        self.V = np.zeros(self.nnodes)

        for element in self.mesh_connectivity:
            # Compute area / volume
            val = 1. / len(element)

            # Define area of element
            if (self.dimensions == 2):
                xi, yi, *_ = self.coords[element[0]]
                xj, yj, *_ = self.coords[element[1]]
                xk, yk, *_ = self.coords[element[2]]
                val *= 0.5 * ((xj - xi) * (yk - yi) - (xk - xi) * (yj - yi))

            self.V[element] += val

    def _set_connectivity(self, initial_crack):
        """
        Sets the sparse connectivity matrix, should only ever be called once.

        :arg initial_crack: The initial crack of the system. The argument may
            be a list of tuples where each tuple is a pair of integers
            representing nodes between which to create a crack. Alternatively,
            the arugment may be a function which takes the (nnodes, 3)
            :class:`numpy.ndarray` of coordinates as an argument, and returns a
            list of tuples defining the initial crack.
        :type initial_crack: list(tuple(int, int)) or function

        :returns: None
        :rtype: NoneType
        """
        if callable(initial_crack):
            initial_crack = initial_crack(self.coords)

        # Calculate the Euclidean distance between each pair of nodes
        distance = cdist(self.coords, self.coords, 'euclidean')

        # Construct the neighbourhood matrix (neighbourhood[i, j] = 1 if i and
        # j are neighbours)
        nnodes = self.nnodes
        neighbourhood = np.zeros((nnodes, nnodes))
        # Connect nodes which are within horizon of each other
        neighbourhood[distance < self.horizon] = 1

        # Construct the initial connectivity matrix
        conn = neighbourhood.copy()
        for i, j in initial_crack:
            # Connectivity is symmetric
            conn[i, j] = 0
            conn[j, i] = 0
        # Nodes are not connected with themselves
        np.fill_diagonal(conn, 0)

        # Initial bond damages
        count = np.sum(conn, axis=0)
        self.family = np.sum(neighbourhood, axis=0)
        damage = np.divide((self.family - count), self.family)
        damage.resize(self.nnodes)

        # Lower triangular - count bonds only once
        # make diagonal values 0
        conn = np.tril(conn, -1)

        # Convert to sparse matrix
        self.connectivity = sparse.csr_matrix(conn)
        self.neighbourhood = sparse.csr_matrix(neighbourhood)

        return damage

    def _set_H(self):
        """
        Constructs the failure strains matrix and H matrix, which is a sparse
        matrix containing distances.

        :returns: None
        :rtype: NoneType
        """
        coords = self.coords

        # Extract the coordinates
        V_x = coords[:, 0]
        V_y = coords[:, 1]
        V_z = coords[:, 2]

        # Tiled matrices
        lam_x = np.tile(V_x, (self.nnodes, 1))
        lam_y = np.tile(V_y, (self.nnodes, 1))
        lam_z = np.tile(V_z, (self.nnodes, 1))

        # Dense matrices
        H_x0 = -lam_x + lam_x.transpose()
        H_y0 = -lam_y + lam_y.transpose()
        H_z0 = -lam_z + lam_z.transpose()

        # Into sparse matrices
        self.H_x0 = sparse.csr_matrix(self.neighbourhood.multiply(H_x0))
        self.H_y0 = sparse.csr_matrix(self.neighbourhood.multiply(H_y0))
        self.H_z0 = sparse.csr_matrix(self.neighbourhood.multiply(H_z0))
        self.H_x0.eliminate_zeros()
        self.H_y0.eliminate_zeros()
        self.H_z0.eliminate_zeros()

        norms_matrix = (
            self.H_x0.power(2) + self.H_y0.power(2) + self.H_z0.power(2)
            )
        self.L_0 = norms_matrix.sqrt()

        # initiate fail_stretches matrix as a linked list format
        fail_strains = np.full((self.nnodes, self.nnodes),
                               self.critical_strain)
        # Make into a sparse matrix
        self.fail_strains = sparse.csr_matrix(fail_strains)

    def bond_stretch(self, u):
        """
        Calculates the strain (bond stretch) of all nodes for a given
        displacement.

        :arg u: The displacement array with shape
            (`nnodes`, `dimension`).
        :type u: :class:`numpy.ndarray`

        :returns: None
        :rtype: NoneType
        """
        cols, rows, data_x, data_y, data_z = [], [], [], [], []

        for i in range(self.nnodes):
            row = self.neighbourhood.getrow(i)

            rows.extend(row.indices)
            cols.extend(np.full((row.nnz), i))
            data_x.extend(np.full((row.nnz), u[i, 0]))
            data_y.extend(np.full((row.nnz), u[i, 1]))
            data_z.extend(np.full((row.nnz), u[i, 2]))

        # Must not be lower triangular
        lam_x = sparse.csr_matrix((data_x, (rows, cols)),
                                  shape=(self.nnodes, self.nnodes))
        lam_y = sparse.csr_matrix((data_y, (rows, cols)),
                                  shape=(self.nnodes, self.nnodes))
        lam_z = sparse.csr_matrix((data_z, (rows, cols)),
                                  shape=(self.nnodes, self.nnodes))

        delH_x = -lam_x + lam_x.transpose()
        delH_y = -lam_y + lam_y.transpose()
        delH_z = -lam_z + lam_z.transpose()

        # Sparse matrices
        self.H_x = delH_x + self.H_x0
        self.H_y = delH_y + self.H_y0
        self.H_z = delH_z + self.H_z0

        norms_matrix = (
            self.H_x.power(2) + self.H_y.power(2) + self.H_z.power(2)
            )

        self.L = norms_matrix.sqrt()

        del_L = self.L - self.L_0

        # Floor values close to zero from del_L sparse matrix
        del_L = del_L.tolil()
        del_L[~(del_L >= 1e-12).toarray()] = 0
        del_L = del_L.tocsr()

        # Step 1. initiate as a sparse matrix
        strain = sparse.lil_matrix(self.connectivity.shape)

        # Step 2. elementwise division
        strain[self.L_0.nonzero()] = (
            del_L[self.L_0.nonzero()]/self.L_0[self.L_0.nonzero()]
            )

        self.strain = strain

    def damage(self):
        """
        Calculates bond damage.

        :returns: A (`nnodes`, ) array containing the damage
            for each node.
        :rtype: :class:`numpy.ndarray`
        """
        # Make sure only calculating for bonds that exist

        # Step 1. initiate as sparse matrix
        bond_healths = sparse.lil_matrix(self.connectivity.shape)

        # Step 2. Find broken bonds, squared as strains can be negative
        bond_healths[self.connectivity.nonzero()] = (
                self.fail_strains.power(2)[self.connectivity.nonzero()]
                - self.strain.power(2)[self.connectivity.nonzero()]
                )

        # Update failed bonds
        bond_healths = bond_healths > 0

        self.connectivity = sparse.csr_matrix(bond_healths)

        # Bond damages
        # Using lower triangular connectivity matrix, so just mirror it for
        # bond damage calc
        temp = self.connectivity + self.connectivity.transpose()

        count = temp.sum(axis=0)
        damage = np.divide((self.family - count), self.family)
        damage.resize(self.nnodes)

        return damage

    def bond_force(self):
        """
        Calculate the force due to bonds acting on each node.

        :returns: A (`nnodes`, 3) array of the component of the force in each
            dimension for each node.
        :rtype: :class:`numpy.ndarray`
        """
        # Container for the forces on each particle in each dimension
        F = np.zeros((self.nnodes, 3))

        # Step 1. Initiate container as a sparse matrix, only need calculate
        # for bonds that exist
        force_normd = sparse.lil_matrix(self.connectivity.shape)

        # Step 2. find normalised forces
        force_normd[self.connectivity.nonzero()] = (
                self.strain[self.connectivity.nonzero()]
                / self.L[self.connectivity.nonzero()]
                )

        # Make lower triangular into full matrix
        force_normd.tocsr()
        force_normd = force_normd + force_normd.transpose()

        # Multiply by the direction and scale of each bond (just trigonometry,
        # we have already scaled for bond length in step 2)
        bond_force_x = force_normd.multiply(self.H_x)
        bond_force_y = force_normd.multiply(self.H_y)
        bond_force_z = force_normd.multiply(self.H_z)

        # now sum along the rows to calculate resultant force on nodes
        F_x = np.array(bond_force_x.sum(axis=0))
        F_y = np.array(bond_force_y.sum(axis=0))
        F_z = np.array(bond_force_z.sum(axis=0))

        F_x.resize(self.nnodes)
        F_y.resize(self.nnodes)
        F_z.resize(self.nnodes)

        # Finally multiply by volume and stiffness
        F_x = self.bond_stiffness * np.multiply(F_x, self.V)
        F_y = self.bond_stiffness * np.multiply(F_y, self.V)
        F_z = self.bond_stiffness * np.multiply(F_z, self.V)

        F[:, 0] = F_x
        F[:, 1] = F_y
        F[:, 2] = F_z

        return F

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
            # Calculate bond stretch, damage and forces on nodes
            self.bond_stretch(u)
            damage = self.damage()
            f = self.bond_force()

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

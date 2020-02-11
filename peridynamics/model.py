from collections import namedtuple
import meshio
import numpy as np
from scipy import sparse
from scipy.spatial.distance import cdist
import warnings


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
        >>> model = Model(
        >>>     mesh_file="./example.msh",
        >>>     horizon=0.1,
        >>>     critical_strain=0.005,
        >>>     elastic_modulus=0.05
        >>>     )
    """
    def __init__(self, mesh_file, horizon, critical_strain, elastic_modulus,
                 initial_crack=None, dimensions=2):
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
        self.set_volume()

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
        self.connectivity = mesh.cells[self.mesh_elements.connectivity]

        # Get boundary connectivity, mesh lines
        self.connectivity_bnd = mesh.cells[self.mesh_elements.boundary]

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
                self.mesh_elements.connectivity: self.connectivity,
                self.mesh_elements.boundary: self.connectivity_bnd
                },
            point_data={
                "damage": damage,
                "displacements": displacements
                },
            file_format=file_format
            )

    def set_volume(self):
        """
        Calculate the value of each node.

        :returns: None
        :rtype: NoneType
        """
        self.V = np.zeros(self.nnodes)

        for element in self.connectivity:
            # Compute area / volume
            val = 1. / len(element)

            # Define area of element
            if (self.dimensions == 2):
                xi, yi, *_ = self.coords[element[0]]
                xj, yj, *_ = self.coords[element[1]]
                xk, yk, *_ = self.coords[element[2]]
                val *= 0.5 * ((xj - xi) * (yk - yi) - (xk - xi) * (yj - yi))

            self.V[element] += val

    def set_connectivity(self):
        """
        Sets the sparse connectivity matrix, should only ever be called once.

        :returns: None
        :rtype: NoneType
        """
        # Initiate connectivity matrix as non sparse
        conn = np.zeros((self.nnodes, self.nnodes))

        # Initiate uncracked connectivity matrix
        conn_0 = np.zeros((self.nnodes, self.nnodes))

        # Check if nodes are connected
        distance = cdist(self.coords, self.coords, 'euclidean')
        for i in range(0, self.nnodes):
            for j in range(0, self.nnodes):
                if distance[i, j] < self.horizon:
                    conn_0[i, j] = 1
                    if i == j:
                        # do not fill diagonal
                        continue
                    elif (not
                          self.is_crack(self.coords[i, :], self.coords[j, :])):
                        conn[i, j] = 1

        # Initial bond damages
        count = np.sum(conn, axis=0)
        self.family = np.sum(conn_0, axis=0)
        damage = np.divide((self.family - count), self.family)
        damage.resize(self.nnodes)

        # Lower triangular - count bonds only once
        # make diagonal values 0
        conn = np.tril(conn, -1)

        # Convert to sparse matrix
        self.conn = sparse.csr_matrix(conn)
        self.conn_0 = sparse.csr_matrix(conn_0)

        return damage

    def set_H(self):
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
        self.H_x0 = sparse.csr_matrix(self.conn_0.multiply(H_x0))
        self.H_y0 = sparse.csr_matrix(self.conn_0.multiply(H_y0))
        self.H_z0 = sparse.csr_matrix(self.conn_0.multiply(H_z0))
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
            row = self.conn_0.getrow(i)

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
        strain = sparse.lil_matrix(self.conn.shape)

        # Step 2. elementwise division
        strain[self.L_0.nonzero()] = (
            del_L[self.L_0.nonzero()]/self.L_0[self.L_0.nonzero()]
            )

        self.strain = strain

        if strain.shape != self.L_0.shape:
            warnings.warn(
                'strain.shape was {}, whilst L_0.shape was {}'.format(
                    strain.shape, self.L_0.shape
                    )
                )

    def damage(self):
        """
        Calculates bond damage.

        :returns: A (`nnodes`, ) array containing the damage
            for each node.
        :rtype: :class:`numpy.ndarray`
        """
        # Make sure only calculating for bonds that exist

        # Step 1. initiate as sparse matrix
        bond_healths = sparse.lil_matrix(self.conn.shape)

        # Step 2. Find broken bonds, squared as strains can be negative
        bond_healths[self.conn.nonzero()] = (
                self.fail_strains.power(2)[self.conn.nonzero()]
                - self.strain.power(2)[self.conn.nonzero()]
                )

        # Update failed bonds
        bond_healths = bond_healths > 0

        self.conn = sparse.csr_matrix(bond_healths)

        # Bond damages
        # Using lower triangular connectivity matrix, so just mirror it for
        # bond damage calc
        temp = self.conn + self.conn.transpose()

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
        force_normd = sparse.lil_matrix(self.conn.shape)

        # Step 2. find normalised forces
        force_normd[self.conn.nonzero()] = (
                self.strain[self.conn.nonzero()]/self.L[self.conn.nonzero()]
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

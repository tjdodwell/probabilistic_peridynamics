from . import periFunctions as func
from collections import namedtuple
import meshio
import numpy as np
from scipy import sparse
import warnings


_MeshElements = namedtuple("MeshElements", ["connectivity", "boundary"])
_mesh_elements_2d = _MeshElements(connectivity="triangle",
                                  boundary="line")
_mesh_elements_3d = _MeshElements(connectivity="tetrahedron",
                                  boundary="triangle")


class SeqModel:
    def __init__(self, dimensions=2):
        self.dimensions = dimensions

        if dimensions == 2:
            self.mesh_elements = _mesh_elements_2d
        elif dimensions == 3:
            self.mesh_elements = _mesh_elements_3d
        else:
            raise DimensionalityError(dimensions)

        # Material Parameters from classical material model
        self.horizon = 0.1
        self.kscalar = 0.05
        self.s00 = 0.05

        self.c = 18.0 * self.kscalar / (np.pi * (self.horizon**4))

    def read_mesh(self, mesh_file):
        mesh = meshio.read(mesh_file)

        # Get coordinates, encoded as mesh points
        self.coords = mesh.points
        self.nnodes = self.coords.shape[0]

        # Get connectivity, mesh triangle cells
        self.connectivity = mesh.cells[self.mesh_elements.connectivity]
        self.nelem = self.connectivity.shape[0]

        # Get boundary connectivity, mesh lines
        self.connectivity_bnd = mesh.cells[self.mesh_elements.boundary]
        self.nelem_bnd = self.connectivity_bnd.shape[0]

    def write_mesh(self, filename, damage=None, displacements=None,
                   file_format=None):
        """
        Write the model's nodes, connectivity and boundary to a mesh file. Also
        write damage and displacements as points data.

        :arg str filename: Path of the file to write the mesh to.
        :arg array optional damage: The damage of each node. Default is None.
        :arg array optional displacments: An array with shape (nnodes, dim)
            where each row is the displacment of a node. Default is None.
        :arg str optional file_format: The file format of the mesh file to
            write. Infered from ``filename`` if None. Default is None.
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

    def setVolume(self):
        self.V = np.zeros(self.nnodes)

        for ie in range(0, self.nelem):
            n = self.connectivity[ie]

            # Compute Area / Volume
            val = 1. / n.size

            # Define area of element
            if (self.dimensions == 2):
                xi = self.coords[int(n[0])][0]
                yi = self.coords[int(n[0])][1]
                xj = self.coords[int(n[1])][0]
                yj = self.coords[int(n[1])][1]
                xk = self.coords[int(n[2])][0]
                yk = self.coords[int(n[2])][1]

                val *= 0.5 * ((xj - xi) * (yk - yi) - (xk - xi) * (yj - yi))

            for j in range(0, n.size):
                self.V[int(n[j])] += val

    def setConn(self, horizon):
        """
        Sets the sparse connectivity matrix, should only ever be called once
        """
        # Initiate connectivity matrix as non sparse
        conn = np.zeros((self.nnodes, self.nnodes))

        # Initiate uncracked connectivity matrix
        conn_0 = np.zeros((self.nnodes, self.nnodes))

        # Check if nodes are connected
        for i in range(0, self.nnodes):
            for j in range(0, self.nnodes):
                if func.l2(self.coords[i, :], self.coords[j, :]) < horizon:
                    conn_0[i, j] = 1
                    if i == j:
                        # do not fill diagonal
                        continue
                    elif (not
                          self.isCrack(self.coords[i, :], self.coords[j, :])):
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

    def setH(self):
        """
        Constructs the covariance matrix, K, failure strains matrix and H
        matrix, which is a sparse matrix containing distances
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

        norms_matrix = (
            np.power(H_x0, 2) + np.power(H_y0, 2) + np.power(H_z0, 2)
            )
        self.L_0 = np.sqrt(norms_matrix)

        # Into sparse matrices
        self.H_x0 = sparse.csr_matrix(self.conn_0.multiply(H_x0))
        self.H_y0 = sparse.csr_matrix(self.conn_0.multiply(H_y0))
        self.H_z0 = sparse.csr_matrix(self.conn_0.multiply(H_z0))
        self.H_x0.eliminate_zeros()
        self.H_y0.eliminate_zeros()
        self.H_z0.eliminate_zeros()

        # Length scale for the covariance matrix
        scale = 0.05

        # Scale of the covariance matrix
        nu = 1e-5

        # inv length scale parameter
        inv_length_scale = (np.divide(-1., 2.*pow(scale, 2)))

        # radial basis functions
        rbf = np.multiply(inv_length_scale, norms_matrix)

        # Exponential of radial basis functions
        K = np.exp(rbf)

        # Multiply by the vertical scale to get covariance matrix, K
        self.K = np.multiply(pow(nu, 2), K)

        # Create L matrix for sampling perturbations
        # epsilon, numerical trick so that M is positive semi definite
        epsilon = 1e-5

        # add epsilon before scaling by a vertical variance scale, nu
        Iden = np.identity(self.nnodes)
        K_tild = K + np.multiply(epsilon, Iden)

        K_tild = np.multiply(pow(nu, 2), K_tild)

        self.C = np.linalg.cholesky(K_tild)
        norms_matrix = sparse.csr_matrix(self.H_x0.power(2)
                                         + self.H_y0.power(2)
                                         + self.H_z0.power(2))
        self.L_0 = norms_matrix.sqrt()

        if (self.H_x0.shape != self.H_y0.shape
                or self.H_x0.shape != self.H_z0.shape):
            raise Exception(
                'The sizes of H_x0, H_y0 and H_z0 did not match!'
                ' The sizes were {}, {}, {}, respectively'.format(
                    self.H_x0.shape, self.H_y0.shape, self.H_z0.shape
                    )
                )

        # initiate fail_stretches matrix as a linked list format
        self.fail_strains = np.full((self.nnodes, self.nnodes), self.s00)
        # Make into a sparse matrix
        self.fail_strains = sparse.csr_matrix(self.fail_strains)

    def calcBondStretchNew(self, U):

        cols, rows, data_x, data_y, data_z = [], [], [], [], []

        for i in range(self.nnodes):
            row = self.conn_0.getrow(i)

            rows.extend(row.indices)
            cols.extend(np.full((row.nnz), i))
            data_x.extend(np.full((row.nnz), U[i, 0]))
            data_y.extend(np.full((row.nnz), U[i, 1]))
            data_z.extend(np.full((row.nnz), U[i, 2]))

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

        # Prune values close to zero from del_L sparse matrix
        del_L[~(del_L >= 1e-12).toarray()] = 0
        del_L.eliminate_zeros()

        # Step 1. initiate as a sparse matrix
        strain = sparse.csr_matrix(self.conn.shape)

        # Step 2. elementwise division
        strain[self.L_0.nonzero()] = sparse.csr_matrix(
            del_L[self.L_0.nonzero()]/self.L_0[self.L_0.nonzero()]
            )

        self.strain = sparse.csr_matrix(strain)
        self.strain.eliminate_zeros()

        if strain.shape != self.L_0.shape:
            warnings.warn(
                'strain.shape was {}, whilst L_0.shape was {}'.format(
                    strain.shape, self.L_0.shape
                    )
                )

    def checkBonds(self):
        """ Calculates bond damage
        """
        # Make sure only calculating for bonds that exist

        # Step 1. initiate as sparse matrix
        bond_healths = sparse.csr_matrix(self.conn.shape)

        # Step 2. Find broken bonds, squared as strains can be negative
        bond_healths[self.conn.nonzero()] = sparse.csr_matrix(
                self.fail_strains.power(2)[self.conn.nonzero()]
                - self.strain.power(2)[self.conn.nonzero()]
                )

        # Update failed bonds
        bond_healths = bond_healths > 0

        self.conn = sparse.csr_matrix(bond_healths)
        self.conn.eliminate_zeros()

        # Bond damages
        # Using lower triangular connectivity matrix, so just mirror it for
        # bond damage calc
        temp = self.conn + self.conn.transpose()

        count = temp.sum(axis=0)
        damage = np.divide((self.family - count), self.family)
        damage.resize(self.nnodes)

        return damage

    def computebondForce(self):
        self.c = 18.0 * self.kscalar / (np.pi * (self.horizon**4))
        # Container for the forces on each particle in each dimension
        F = np.zeros((self.nnodes, 3))

        # Step 1. Initiate container as a sparse matrix, only need calculate
        # for bonds that exist
        force_normd = sparse.csr_matrix(self.conn.shape)

        # Step 2. find normalised forces
        force_normd[self.conn.nonzero()] = sparse.csr_matrix(
                self.strain[self.conn.nonzero()]/self.L[self.conn.nonzero()]
                )

        # Make lower triangular into full matrix
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
        F_x = self.c * np.multiply(F_x, self.V)
        F_y = self.c * np.multiply(F_y, self.V)
        F_z = self.c * np.multiply(F_z, self.V)

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

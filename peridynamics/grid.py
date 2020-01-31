import numpy as np


class Grid:
    def __init__(self):
        self.dim = 2
        self.elements = 0
        self.nodes = 0

    def build_structured_mesh(self, L, n, X0):
        self.n = n

        # Function builds a structured finite element mesh in 2D
        nodes_x = n[0] + 1
        nodes_y = n[1] + 1
        self.nodePerElement = 4
        self.nodes = nodes_x * nodes_y
        self.elements = n[0] * n[1]

        x = np.linspace(X0[0], X0[0] + L[0], nodes_x)
        y = np.linspace(X0[1], X0[1] + L[1], nodes_y)

        self.h = np.zeros(self.dim)

        for i in range(0, self.dim):
            self.h[i] = L[i] / n[i]

        # Nodes will be formed from a tensor product of this two vectors
        self.coords = np.zeros((self.nodes, 2))

        count = 0
        for i in range(0, n[1] + 1):
            for j in range(0, n[0] + 1):
                self.coords[count][0] = x[j]
                self.coords[count][1] = y[i]
                count += 1

        # Build Connectivity Matrix
        self.elements = n[0] * n[1]
        self.connectivity = np.zeros((self.elements, self.nodePerElement),
                                     dtype=int)
        count = 0
        ncount = 0
        for j in range(0, n[1]):
            for i in range(0, n[0]):
                self.connectivity[count][0] = ncount
                self.connectivity[count][1] = ncount + 1
                self.connectivity[count][2] = ncount + nodes_x + 1
                self.connectivity[count][3] = ncount + nodes_x
                count += 1
                ncount += 1
            ncount += 1

    def particle_to_cell(self, coords_particles):
        particles = int(coords_particles[:].size / self.dim)
        p2e = np.zeros(particles, dtype=int)
        coords_local = np.zeros((particles, self.dim))

        # For each of the particles
        for i in range(0, particles):
            # particle coordinates
            xP = coords_particles[i][:]
            id_ = np.zeros(self.dim)
            for j in range(0, self.dim):
                id_[j] = np.floor(xP[j] / self.h[j])
                # Catch boundary case
                if id_[j] == self.n[j]:
                    id_[j] -= 1
            if self.dim == 2:
                p2e[i] = self.n[0] * id_[1] + id_[0]
            else:
                p2e[i] = (self.n[0] * self.n[1] * id_[1]
                          + self.n[0] * id_[1] + id_[0])
            # Global to local mapping is easy as structured grid / domain
            node = self.connectivity[p2e[i]][:]
            for j in range(0, self.dim):
                coords_local[i][j] = (
                    (2 / self.h[j])
                    * (coords_particles[i][j]
                       - (self.coords[node[0]][j] + 0.5 * self.h[j]))
                    )

        return coords_local, p2e

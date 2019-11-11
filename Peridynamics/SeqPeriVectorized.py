
import numpy as np

import PeriParticle as peri

import periFunctions as func

from scipy import sparse

import time

class SeqModel:
	def __init__(self):
        ## Scalars
		# self.nnodes defined when instance of readMesh called, cannot initialise any other matrix until we know the nnodes

		self.dim = 2

		self.meshFileName = "test.msh"

		self.meshType = 2
		self.boundaryType = 1
		self.numBoundaryNodes = 2
		self.numMeshNodes = 3

		# Material Parameters from classical material model
		self.horizon = 0.1
		self.K = 0.05
		self.s00 = 0.05

		self.c = 18.0 * self.K / (np.pi * (self.horizon**4));


		if (self.dim == 3):
			self.meshType = 4
			self.boundaryType = 2
			self.numBoundaryNodes = 3
			self.numMeshNodes = 4

	
	def readMesh(self, fileName):

		f = open(fileName, "r")

		if f.mode == "r":

			iline = 0

			# Read the Nodes in the Mesh First

			findNodes = 0
			while (findNodes == 0):
				iline += 1
				line = f.readline()
				if(line.strip() == '$Nodes'):
					findNodes = 1

			line = f.readline()
			self.nnodes = int(line.strip())
			self.coords = np.zeros((self.nnodes, 3), dtype=np.float64)

			for i in range(0, self.nnodes):
				iline += 1
				line = f.readline()
				rowAsList = line.split()
				self.coords[i][0] = rowAsList[1]
				self.coords[i][1] = rowAsList[2]
				self.coords[i][2] = rowAsList[3]

			line = f.readline() # This line will read $EndNodes - Could add assert on this
			line = f.readline() # This line will read $Elements

			# Read the Elements from the mesh for the volume calculations connectivity 

			line = f.readline() # This gives the total number of elements - but includes all types of elements
			self.totalNel = int(line.strip())
			self.connectivity = []
			self.connectivity_bnd = []


			for ie in range(0, self.totalNel):

				iline +=1
				line = f.readline()
				rowAsList = line.split()

				if(int(rowAsList[1]) == self.boundaryType):
					tmp = np.zeros(self.dim)
					for k in range(0, self.dim):
						tmp[k] = int(rowAsList[5 + k]) - 1
					self.connectivity_bnd.append(tmp)

				elif(int(rowAsList[1]) == self.meshType):
					tmp = np.zeros(self.dim + 1)
					for k in range(0, self.dim + 1):
						tmp[k] = int(rowAsList[5 + k]) - 1
					self.connectivity.append(tmp)

			self.nelem = len(self.connectivity)

			self.nelem_bnd = len(self.connectivity_bnd)

		f.close()
		
	def setVolume(self):

		self.V = np.zeros(self.nnodes)

		for ie in range(0,self.nelem):

			n = self.connectivity[ie]

			# Compute Area / Volume

			val = 1. / n.size

			if (self.dim == 2): # Define area of element

				xi = self.coords[int(n[0])][0]
				yi = self.coords[int(n[0])][1]
				xj = self.coords[int(n[1])][0]
				yj = self.coords[int(n[1])][1]
				xk = self.coords[int(n[2])][0]
				yk = self.coords[int(n[2])][1]

				val *= 0.5 * ( (xj - xi) * (yk - yi) - (xk - xi) * (yj - yi) )

			for j in range(0,n.size):
				self.V[int(n[j])] += val
				

	def setConn(self, horizon):
		""" sets the sparse connectivity matrix, should only ever be called once
		"""
		# Initiate connectivity matrix as non sparse
		conn = np.zeros((self.nnodes, self.nnodes))
		
		# Initiate uncracked connectivity matrix
		conn_0 = np.zeros((self.nnodes, self.nnodes))
		
		# Check if nodes are connected
		for i in range(0, self.nnodes):		
			for j in range(0, self.nnodes):
				if(func.l2(self.coords[i,:], self.coords[j,:]) < horizon):
					conn_0[i, j] = 1
					if (self.isCrack(self.coords[i,:], self.coords[j,:]) == False):
						conn[i, j] = 1
						
					
		# Initial bond damages
		count = np.sum(conn, axis =0)
		family = np.sum(conn_0, axis=0)
		damage = np.divide((family - count), family)			
						
		# Convert to sparse matrix
		self.conn = sparse.csr_matrix(conn)
		
		# intial connectivity matrix
		self.conn_0 = sparse.csr_matrix(conn_0)
		
		if damage.shape != (self.nnodes, ):
			raise Exception('damage vector must have shape {}. damage had shape {}'.format((self.nnodes, ),(damage.shape)))
		
		return damage
	
	def setH(self):
		""" Constructs the covariance matrix, K, failure strains matrix and H matrix, which is a sparse matrix containing distances
		"""
		st = time.time()
		coords = self.coords
		
		
		V_x = coords[:,0]
		lam_x = np.tile(V_x, (self.nnodes,1 ))
		del V_x
		
		V_y = coords[:, 1]
		lam_y = np.tile(V_y, (self.nnodes, 1))
		del V_y
		
		V_z = coords[:, 1]
		lam_z = np.tile(V_z, (self.nnodes, 1))
		del V_z
		
		H_x0 = lam_x - lam_x.transpose()
		H_y0 = lam_y - lam_y.transpose()
		H_z0 = lam_z - lam_y.transpose()
		
		norms_matrix = np.power(H_x0, 2) + np.power(H_y0, 2) + np.power(H_z0, 2)
		print('the type and shape of norms_matrix are {} and {}'.format(type(norms_matrix), norms_matrix.shape))
		
		
# =============================================================================
# 		# TODO: Should I initiate connectivity matrices here instead?
# 		# Initiate is crack matrix
# 		is_crack_matrix = np.zeros((self.nnodes, self.nnodes))
# 		
# 		# Check if cracked, can this be vectorized?
# 		for i in range(0, self.nnodes):		
# 			for j in range(0, self.nnodes):
# 				if (self.isCrack(self.coords[i,:], self.coords[j,:]) == True):
# 					is_crack_matrix[i, j] = 1
# 		
# 		is_crack_matrix = is_crack_matrix.astype(bool)
# 		
# 		# connectivity matrix initialization
# 		horizon_matrix = np.full((self.nnodes, self.nnodes), pow(self.horizon, 2))
# 		conn_0_matrix = horizon_matrix - norms_matrix
# 		conn_0_matrix[conn_0_matrix < 0] = 0
# 		conn_0_matrix = conn_0_matrix.astype(bool)
# 		
# 		conn_0_matrix = 
# 		# connectivity matrix post crack defect intialisation
# 		conn_matrix
# 		
# 		
# 		conn_0_matrix = sparse.csr_matrix(conn_matrix)
# 		conn_0_matrix.eliminate_zeros()
# 		self.conn_0 = conn_matrix.prune() #need to prune?
# 		print('conn', self.conn_0)
# 		print('shape conn', self.conn.shape)
# =============================================================================
		
		
		# Length scale for the covariance matrix
		l = 0.001
		
		# Scale of the covariance matrix
		nu = 0.1
		
		# inv length scale parameter
		inv_length_scale = (np.divide(-1., 2.*pow(l, 2)))
		
		# radial basis functions
		rbf = np.multiply(inv_length_scale, norms_matrix)
		
		# Exponential of radial basis functions
		K = np.exp(rbf)
		
		# Multiply by the vertical scale to get covariance matrix, K
		self.K = np.multiply(pow(nu, 2), K)
		
		
		self.H_x0 = sparse.csr_matrix(self.conn.multiply(H_x0))
		self.H_y0 = sparse.csr_matrix(self.conn.multiply(H_y0))
		self.H_z0 = sparse.csr_matrix(self.conn.multiply(H_z0))
		
		# initiate fail_stretches matrix as a linked list format
		fail_strains = np.full((self.nnodes, self.nnodes), self.s00)
		
		fail_strains = sparse.csr_matrix(self.conn.multiply(fail_strains))
		fail_strains.eliminate_zeros()
		self.fail_strains = fail_strains.prune()
		
		print('Hello')
		
		print('Type of fail strains is {}'.format(type(self.fail_strains)))
		
		print('Constructed H in {} seconds'.format(time.time() - st))

	def calcBondStretch(self, U):
		
		st = time.time()
		
		shape = (self.nnodes, self.nnodes)
		
		H_x0, H_y0, H_z0 = self.H_x0, self.H_y0, self.H_z0
		
        	
		delV_x = U[:, 0]
		lam_x = np.tile(delV_x, (self.nnodes, 1))
	
		delV_y = U[:, 1]
		lam_y = np.tile(delV_y, (self.nnodes, 1))
		
		delV_z = U[:, 2]
		lam_z = np.tile(delV_z, (self.nnodes, 1))
		
		
		delH_x = sparse.csr_matrix(self.conn.multiply(lam_x) - self.conn.multiply(lam_x.transpose()))
		delH_y = sparse.csr_matrix(self.conn.multiply(lam_y) - self.conn.multiply(lam_y.transpose()))
		delH_z = sparse.csr_matrix(self.conn.multiply(lam_z) - self.conn.multiply(lam_z.transpose()))
		
		print('the type of delH_x is {} and the shape is {}'.format(type(delH_x), delH_x.shape))
		
		tmp_x = 1 / H_x0.d
		
		tmp_x = sparse.csr_matrix(delH_x[H_x0.nonzero()] / H_x0[H_x0.nonzero()])
		tmp_y = sparse.csr_matrix(delH_y[H_y0.nonzero()] / H_y0[H_y0.nonzero()])
		tmp_z = sparse.csr_matrix(delH_z[H_z0.nonzero()] / H_z0[H_z0.nonzero()])
		
		
		print('the type of tmp_x is {} and the shape is {}'.format(type(tmp_x), tmp_x.shape))
		print('temp_x')
		
		# TODO: fix bug where reshape shape is not compatible with sparse matrix
		tmp_x.resize(shape)
		tmp_y.resize(shape)
		tmp_z.resize(shape)
		
		self.H_x = H_x0 + delH_x
		self.H_y = H_y0 + delH_y
		self.H_z = H_z0 + delH_z
		
		tmp = tmp_x.power(2) + tmp_y.power(2) + tmp_z.power(2)
		print('the type of tmp is {} and the shape is {}'.format(type(tmp), tmp.shape))
		self.strain = tmp.sqrt()
		print('the type of strain is {} and the shape is {}'.format(type(self.strain), self.strain.shape))

		print('Constructed H in {} seconds'.format(time.time() - st))
		
	def checkBonds(self):
		""" Calculates bond damage
		"""
		# if bond_health value is less than 0 then it is a broken bond
		bond_healths = self.fail_strains - self.strain.sign().multiply(self.strain)
		bond_healths[bond_healths < 0] = 0
		bond_healths = bond_healths.astype(bool)
		
		# connectivity matrix
		bond_healths.eliminate_zeros()
		self.conn = bond_healths.prune() #need to prune?
		print(self.conn)
		print(self.conn.shape)
		
		# bond damages
		count = self.conn.sum(axis = 0)
		family = self.conn_0.sum(axis = 0)
		damage = np.divide((family - count), family)
		
		if damage.shape != (self.nnodes, ):
			raise Exception('damage vector must have shape {}. damage had shape {}'.format((self.nnodes, ),(damage.shape)))
		
		return damage
		
		
	def computebondForce(self):
		
		self.c = 18.0 * self.K / (np.pi * (self.horizon**4))
		F = np.zeros((self.nnodes,3)) # Container for the forces on each particle in each dimension
		
		force_mags = self.strain.multiply(self.V)
		force_mags =  force_mags.multiply(self.c)
		
		force_norms = sparse.csr_matrix(force_mags[self.norms.nonzero()]/ self.norms[self.norms.nonzero()])
		
		# norms for the bond lengths
		norms = self.H_x.power(2) + self.H_y.power(2) + self.H_z.power(2)
		norms = norms.sqrt()
		
		bond_force_x = sparse.csr_matrix(force_norms[norms.nonzero()].multiply(self.H_x[norms.nonzero()]))
		bond_force_y = sparse.csr_matrix(force_norms[norms.nonzero()].multiply(self.H_x[norms.nonzero()]))
		bond_force_z = sparse.csr_matrix(force_norms[norms.nonzero()].multiply(self.H_x[norms.nonzero()]))
		
		
		F_x = bond_force_x.sum(axis = 0)
		F_y = bond_force_y.sum(axis = 0)
		F_z = bond_force_z.sum(axis = 0)
		F[:, 0] = F_x.todense()
		F[:, 1] = F_y.todense()
		F[:, 2] = F_z.todense()
		
		assert F.shape == (self.nnodes, 3)
		
		return F

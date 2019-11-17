
import numpy as np

import PeriParticle as peri

import periFunctions as func

from scipy import sparse

import warnings

import time


class SeqModel:
	def __init__(self):
        ## Scalars
		# self.nnodes defined when instance of readMesh called, cannot initialise any other matrix until we know the nnodes
		self.v = False # is this needed here, since it was put in MODEL class, simplesquare?
		self.dim = 2

		self.meshFileName = "test.msh"

		self.meshType = 2
		self.boundaryType = 1
		self.numBoundaryNodes = 2
		self.numMeshNodes = 3

		# Material Parameters from classical material model
		self.horizon = 0.1
		self.kscalar = 0.05
		self.s00 = 0.05

		self.c = 18.0 * self.kscalar / (np.pi * (self.horizon**4));


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
					if i == j:
						pass # do not fill diagonal
					elif (self.isCrack(self.coords[i,:], self.coords[j,:]) == False):
						conn[i, j] = 1
						
					
		# Initial bond damages
		count = np.sum(conn, axis =0)
		self.family = np.sum(conn_0, axis=0)
		damage = np.divide((self.family - count), self.family)
		damage.resize(self.nnodes)
		
		print('initial damage vector is {}'.format(damage))
		
# =============================================================================
# 		# Lower triangular - count bonds only once
# 		# make diagonal values 0
# 		conn = np.tril(conn, -1)
# =============================================================================
		
		# Convert to sparse matrix
		self.conn = sparse.csr_matrix(conn)
		
		if self.v:
			print('self.conn is HERE', self.conn)
		
		# intial connectivity matrix
		#self.conn_0 = sparse.csr_matrix(conn_0)
		

		return damage
	
	def setH(self):
		""" Constructs the covariance matrix, K, failure strains matrix and H matrix, which is a sparse matrix containing distances
		"""
		st = time.time()
		coords = self.coords
		
		
		V_x = coords[:,0]
		lam_x = sparse.csr_matrix(np.tile(V_x, (self.nnodes,1 )))
		
		V_y = coords[:, 1]
		lam_y = sparse.csr_matrix(np.tile(V_y, (self.nnodes, 1)))
		
		V_z = coords[:, 2]
		lam_z = sparse.csr_matrix(np.tile(V_z, (self.nnodes, 1)))
		
				
		H_x0 = sparse.csr_matrix(lam_x.multiply(self.conn) - lam_x.transpose().multiply(self.conn))
		H_y0 = sparse.csr_matrix(lam_y.multiply(self.conn) - lam_y.transpose().multiply(self.conn))
		H_z0 = sparse.csr_matrix(lam_z.multiply(self.conn) - lam_z.transpose().multiply(self.conn))
		
		
		norms_matrix = H_x0.power(2) + H_y0.power(2) + H_z0.power(2)
		
		self.L_0 = norms_matrix.sqrt()
		self.H_x0 = H_x0
		self.H_y0 = H_y0
		self.H_z0 = H_z0
		
		
		#print('the type and shape of norms_matrix are {} and {}'.format(type(norms_matrix), norms_matrix.shape))
		
		
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
		
		
# =============================================================================
# 		# Length scale for the covariance matrix
# 		l = 0.001
# 		
# 		# Scale of the covariance matrix
# 		nu = 0.1
# 		
# 		# inv length scale parameter
# 		inv_length_scale = (np.divide(-1., 2.*pow(l, 2)))
# 		
# 		# radial basis functions
# 		rbf = np.multiply(inv_length_scale, norms_matrix)
# 		
# 		# Exponential of radial basis functions
# 		K = np.exp(rbf)
# 		
# 		# Multiply by the vertical scale to get covariance matrix, K
# 		self.K = np.multiply(pow(nu, 2), K)
# =============================================================================
		
		

		if self.H_x0.shape != self.H_y0.shape or self.H_x0.shape != self.H_z0.shape:
			raise Exception(' The sizes of H_x0, H_y0 and H_z0 did not match! The sizes were {}, {}, {}, respectively'.format(self.H_x0.shape, self.H_y0.shape, self.H_z0.shape))
		

		if self.v:
			print(self.L_0, self.L_0.shape, 'here is L_0')
		#self.L_0 = self.H_x0.power(2) + self.H_y0.power(2) + self.H_z0.power(2) # no need to recalculate
        
        # TODO: resize L_0 to same size as H_x0?
		
		if self.L_0.shape != self.H_x0.shape:
			print(' The size of the connectivity matrix is {}'.format(self.conn.shape))
			warnings.warn('L_0.size was {}, whilst H_x0.size was {}, they should be the same size'.format(self.L_0.shape, self.H_x0.shape))
		
		# initiate fail_stretches matrix as a linked list format
		self.fail_strains = np.full((self.nnodes, self.nnodes), self.s00)
		# Make into a sparse matrix
		self.fail_strains = sparse.csr_matrix(self.fail_strains)
		
		if self.v:
			print('Type of fail strains is {} and the shape is {}'.format(type(self.fail_strains), self.fail_strains.shape))
		
		print('Constructed H in {} seconds'.format(time.time() - st))

	def calcBondStretch(self, U):
		
		st = time.time()
		
		delV_x = U[:, 0]
		lam_x = sparse.csr_matrix(np.tile(delV_x, (self.nnodes, 1)))
	
		delV_y = U[:, 1]
		lam_y = sparse.csr_matrix(np.tile(delV_y, (self.nnodes, 1)))
		
		delV_z = U[:, 2]
		lam_z = sparse.csr_matrix(np.tile(delV_z, (self.nnodes, 1)))
		
		delH_x = sparse.csr_matrix(lam_x.multiply(self.conn) - lam_x.transpose().multiply(self.conn))
		delH_y = sparse.csr_matrix(lam_y.multiply(self.conn) - lam_y.transpose().multiply(self.conn))
		delH_z = sparse.csr_matrix(lam_z.multiply(self.conn) - lam_z.transpose().multiply(self.conn))
		
		self.H_x = delH_x + self.H_x0
		self.H_y = delH_y + self.H_y0
		self.H_z = delH_z + self.H_z0
		
		# Compute bond length matrix
		# bond lengths at current time step
		# Step 1. Initiate as a sparse matrix
			
		norms_matrix = self.H_x.power(2) + self.H_y.power(2) + self.H_z.power(2)
		
		#L = np.power(self.H_x, 2) + np.power(self.H_y, 2) + np.power(self.H_z, 2)
		self.L = norms_matrix.sqrt()
		
		if self.v:
			print(' The shape of L is {}, {}'.format(self.L.shape, self.L))
			
			print(delH_x, 'ABOVE is delH_x')
			
			print(self.H_x, 'ABOVE is H_x')
		
		
		#del_L = delH_x.power(2) + delH_y.power(2) + delH_z.power(2)
		del_L = self.L - self.L_0
		
		# TODO: if any element of del_l < 1e-12 then set it to 0, to kill a rounding error?
		
		print('del_L shape', del_L.shape)
		
		# Step 1. initiate as a sparse matrix
		strain = sparse.csr_matrix(self.conn.shape)
		
		# Step 2. elementwise division
		strain[self.L_0.nonzero()] = sparse.csr_matrix(del_L[self.L_0.nonzero()]/self.L_0[self.L_0.nonzero()])
		
		
		self.strain = strain.multiply(self.conn)
		#self.strain.eliminate_zeros()
		
		if strain.shape != self.L_0.shape:
			warnings.warn('strain.shape was {}, whilst L_0.shape was {}'.format(strain.shape, self.L_0.shape))
		
	
		print('the type of strain is {} and the shape is {}'.format(type(self.strain), self.strain.shape))

		print('Constructed bond strain matrix in {} seconds'.format(time.time() - st))
		
	def checkBonds(self):
		""" Calculates bond damage
		"""
		# if bond_health value is less than 0 then it is a broken bond
		
# =============================================================================
# 		if t == 1:
# 			bond_healths = self.fail_strains
# 			# bond damages
# 			count = self.conn.sum(axis = 0)
# 			family = self.conn_0.sum(axis = 0)
# 			damage = np.divide((family - count), family)[0]
# 			
# 			return damage
# =============================================================================
			
		
		# Make sure only calculating for bonds that exist
		
		# Step 1. initiate as sparse matrix
		bond_healths = sparse.csr_matrix(self.conn.shape)
		
		# Step 2. Find broken bonds
		
		#bond_healths[self.conn.nonzero()] = sparse.csr_matrix(self.fail_strains[self.conn.nonzero()] - self.strain[self.conn.nonzero()])
		
		bond_healths[self.conn.nonzero()] = sparse.csr_matrix(self.fail_strains.power(2)[self.conn.nonzero()] - self.strain.power(2)[self.conn.nonzero()])
		
		bond_healths[bond_healths < 0] = 0
		
		# play around with converting to bool instead
		bond_healths[bond_healths > 0] = 1
		
		bond_healths.eliminate_zeros() #needed?
		self.conn = bond_healths
		#print(self.conn.shape)
		
		# Bond damages
# =============================================================================
# 		
# 		# Using lower triangular connectivity matrix, so just mirror it for bond damage calc
# 		temp = self.conn + self.conn.transpose()
# =============================================================================
		count = self.conn.sum(axis = 0)
		damage = np.divide((self.family - count), self.family)
		damage.resize(self.nnodes)
		
		print(np.max(damage), 'max_damage')
		print(np.min(damage), 'min_damage')

		print(damage, damage.shape)
	
		return damage
		
		
	def computebondForce(self):
		
		self.c = 18.0 * self.kscalar / (np.pi * (self.horizon**4))
		F = np.zeros((self.nnodes,3)) # Container for the forces on each particle in each dimension
		
		# since strain is a lower triangular matrix, mirror it to get all bond strains
# =============================================================================
# 		temp_strain = self.strain + self.strain.transpose() #perhaps negative here instead, TODO: check this
# 		temp_conn = self.conn + self.conn.transpose()
# =============================================================================
		# Then multiply the strain matrix by the rvecs, then divide by normalising terms
		
		# Get the force resolving terms for each bond, in each direction (just trigonometry)
		
		# Step 1. Initiate container as a sparse matrix, only need calculate for bonds that exist
		force_normd = sparse.csr_matrix(self.conn.shape)

		# Step 2. find normalised forces
		#force_normd[self.conn.nonzero()] = sparse.csr_matrix(self.strain[self.conn.nonzero()]/self.L[self.conn.nonzero()])
		force_normd[self.L.nonzero()] = sparse.csr_matrix(self.strain[self.L.nonzero()]/self.L[self.L.nonzero()])
		force_normd = force_normd.multiply(self.conn)
		
		H_x_abs = abs(self.H_x)
		H_y_abs = abs(self.H_y)
		H_z_abs = abs(self.H_z)
		
		
		bond_force_x = force_normd.multiply(H_x_abs)
		bond_force_y = force_normd.multiply(H_y_abs)
		bond_force_z = force_normd.multiply(H_z_abs)
		
		
# =============================================================================
# 		# Make lower triangular into full matrix
# 		bond_force_x = bond_force_x + bond_force_x.transpose()
# 		bond_force_y = bond_force_y + bond_force_y.transpose()
# 		bond_force_z = bond_force_z + bond_force_z.transpose()
# =============================================================================
		
		# now sum along the rows to calculate force on nodes
		F_x = np.array(bond_force_x.sum(axis = 0))
		F_y = np.array(bond_force_y.sum(axis = 0))
		F_z = np.array(bond_force_z.sum(axis = 0))
		
		F_x.resize(self.nnodes)
		F_y.resize(self.nnodes)
		F_z.resize(self.nnodes)
		
		
		# Finally multiply by volume and stiffness
		F_x = self.c * np.multiply(F_x, self.V)
		F_y = self.c * np.multiply(F_y, self.V)
		F_z = self.c * np.multiply(F_z, self.V)
		
		if self.v == 1:
			print(F_x, 'The shape of F_x is', F_x.shape, type(F_x))
			print(self.V, 'The shape of V is', self.V.shape, type(self.V))
		
		F[:, 0] = F_x
		F[:, 1] = F_y
		F[:, 2] = F_z
		
		assert F.shape == (self.nnodes, 3)
	
		return F

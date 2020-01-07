"""
Created on Sun Nov 10 16:25:58 2019

@author: Ben Boys
"""

import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../../Peridynamics')
sys.path.insert(1, '../../PostProcessing')
sys.path.insert(1, '../../FEM')

import PeriParticle as peri
from OpenCLPeriVectorized import SeqModel as MODEL
import numpy as np
import scipy.stats as sp
import vtk as vtk
import time
from timeit import default_timer as timer

import grid as fem

import pyopencl as cl
import sys


class simpleSquare(MODEL):

	# A User defined class for a particular problem which defines all necessary parameters

	def __init__(self):
		
		# verbose
		self.v = True
		# TODO remove dim
		self.dim = 2
		
		self.meshFileName = 'test.msh'

		self.meshType = 2
		self.boundaryType = 1
		self.numBoundaryNodes = 2
		self.numMeshNodes = 3
		
		# Material Parameters from classical material model
		self.PD_HORIZON = 0.1
		self.PD_K = 0.05
		self.PD_S0 = 0.005
		self.PD_E = (18.00 * self.PD_K) / (np.pi * np.power(self.PD_HORIZON, 4))
		
		# User input parameters
		self.loadRate = 0.00001
		self.crackLength = 0.3
		self.dt = 1e-3
		
		# These parameters will eventually be passed to model via command line arguments

		self.readMesh(self.meshFileName)
		
		# No. coordinate dimensions
		self.DPN = 3
		self.PD_DPN_NODE_NO = int(self.DPN * self.nnodes)
		
		st = time.time()
		self.setNetwork(self.PD_HORIZON)
		print("Building horizons took {} seconds. Horizon length: {}".format((time.time() -st), self.MAX_HORIZON_LENGTH))
		#self.setH() #TODO
		self.setVolume()
		
		self.bctypes = np.zeros((self.nnodes, self.DPN))
		self.bcvalues = np.zeros((self.nnodes, self.DPN))
		
		# Find the boundary nodes
		# -1 for LHS and +1 for RHS. 0 for NOT ON BOUNDARY
		for i in range(0, self.nnodes):
			bnd = self.findBoundary(self.coords[i][:])
			self.bctypes[i, 0] = bnd
			self.bctypes[i, 1] = bnd
			self.bctypes[i, 2] = bnd
			self.bcvalues[i, 0] = bnd*0.5 * self.dt * self.loadRate
			self.bcvalues[i, 1] = bnd*0.5 * self.dt * self.loadRate
			self.bcvalues[i, 2] = bnd*0.5 * self.dt * self.loadRate

# =============================================================================
# 		# Build Finite Element Grid Overlaying particles
# 		myGrid = fem.Grid()
# 
# 		self.L = []
# 		self.X0  = [0.0, 0.0] # bottom left
# 		self.nfem = []
# 
# 		for i in range(0, self.dim):
# 			self.L.append(np.max(self.coords[:,i]))
# 			self.nfem.append(int(np.ceil(self.L[i] / self.PD_HORIZON)))
# 
# 		myGrid.buildStructuredMesh2D(self.L,self.nfem,self.X0,1)
# 
# 		self.p_localCoords, self.p2e = myGrid.particletoCell_structured(self.coords[:,:self.dim])
# =============================================================================

	def findBoundary(self,x):
		# Function which markes constrain particles
	    bnd = 0 # Does not live on a boundary
	    if (x[0] < 1.5 * self.PD_HORIZON):
	        bnd = -1
	    elif (x[0] > 1.0 - 1.5 * self.PD_HORIZON):
	        bnd = 1
	    return bnd

	def isCrack(self, x, y):
		output = 0
		p1 = x
		p2 = y
		if(x[0] > y[0]):
			p2 = x
			p1 = y
		if ((p1[0] < 0.5 + 1e-6) and (p2[0] > 0.5 + 1e-6)): # 1e-6 makes it fall one side of central line of particles
		    # draw a straight line between them
			m = (p2[1] - p1[1]) / (p2[0] - p1[0])
			c = p1[1] - m * p1[0]
			height = m * 0.5 + c # height a x = 0.5
			if ((height > 0.5 * (1 - self.crackLength)) and (height < 0.5 * (1 + self.crackLength))):
				output = 1
		return output


def output_device_info(device_id):
    sys.stdout.write("Device is ")
    sys.stdout.write(device_id.name)
    if device_id.type == cl.device_type.GPU:
        sys.stdout.write("GPU from ")
    elif device_id.type == cl.device_type.CPU:
        sys.stdout.write("CPU from ")
    else:
        sys.stdout.write("non CPU of GPU processor from ")
    sys.stdout.write(device_id.vendor)
    sys.stdout.write(" with a max of ")
    sys.stdout.write(str(device_id.max_compute_units))
    sys.stdout.write(" compute units\n")
    sys.stdout.flush()
	
	
def multivar_normal(L, num_nodes):
		""" Fn for taking a single multivar normal sample covariance matrix with Cholesky factor, L
		"""
		zeta = np.random.normal(0, 1, size = num_nodes)
		zeta = np.transpose(zeta)

		w_tild = np.dot(L, zeta) #vector

		return w_tild

def noise(L, samples, num_nodes):
		""" takes multiple samples from multivariate normal distribution with covariance matrix whith Cholesky factor, L
		"""

		noise = []

		for i in range(samples):
			noise.append(multivar_normal(L, num_nodes))

		return np.transpose(noise)


def sim(sample, myModel, numSteps = 400, numSamples = 1, print_every = 10):
	print("Peridynamic Simulation -- Starting")
	
	# Initializing OpenCL
	context = cl.create_some_context()
	queue = cl.CommandQueue(context)
	
	# Print out device info
	output_device_info(context.devices[0])
	
	# Build the OpenCL program from file
	kernelsource = open("opencl_peridynamics.cl").read()
	
	SEP = " "
	
	options_string ="-cl-fast-relaxed-math" + SEP +\
					"-DPD_DPN_NODE_NO=" + str(myModel.PD_DPN_NODE_NO) + SEP +\
					"-DPD_NODE_NO=" + str(myModel.nnodes) + SEP +\
					"-DMAX_HORIZON_LENGTH=" + str(myModel.MAX_HORIZON_LENGTH) + SEP +\
					"-DPD_DT=" + str(myModel.dt) + SEP +\
					"-DPD_E=" + str(myModel.PD_E) + SEP +\
					"-DPD_S0=" + str(myModel.PD_S0)
	
	program = cl.Program(context, kernelsource).build([options_string])
	cl_kernel_initial_values = program.InitialValues
	cl_kernel_time_marching_1 = program.TimeMarching1
	cl_kernel_time_marching_2 = program.TimeMarching2
	cl_kernel_time_marching_3 = program.TimeMarching3
	cl_kernel_check_bonds = program.CheckBonds
	cl_kernel_calculate_damage = program.CalculateDamage
	
	# Set initial values in host memory
	
	# horizons and horizons lengths
	h_horizons = myModel.horizons
	h_horizons_lengths = myModel.horizons_lengths
	print(h_horizons_lengths)
	print(h_horizons)
	print("shape horizons lengths", h_horizons_lengths.shape)
	print("shape horizons lengths", h_horizons.shape)
	
	# Nodal coordinates
	h_coords = myModel.coords
	
	# Boundary conditions types and delta values
	h_bctypes = myModel.bctypes
	h_bcvalues = myModel.bcvalues
	
	print("bctypes", h_bctypes)
	
	# Nodal volumes
	h_vols = myModel.V
	
	# Displacements
	h_un = np.empty((myModel.nnodes, myModel.DPN))
	h_un1 = np.empty((myModel.nnodes, myModel.DPN))
	
	# Forces
	h_udn = np.empty((myModel.nnodes, myModel.DPN))
	h_udn1 = np.empty((myModel.nnodes, myModel.DPN))
	
	# Damage vector
	h_damage = np.empty(myModel.nnodes).astype(np.float32)
	
	# Build OpenCL data structures
	
	# Read only
	d_coords = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_coords)
	d_bctypes = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_bctypes)
	d_bcvalues = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_bcvalues)
	d_vols = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_vols)
	d_horizons_lengths = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_horizons_lengths)
	
	# Read and write
	d_horizons = cl.Buffer(context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_horizons)
	d_un = cl.Buffer(context, cl.mem_flags.READ_WRITE, h_un.nbytes)
	d_udn = cl.Buffer(context, cl.mem_flags.READ_WRITE, h_udn.nbytes)
	d_un1 = cl.Buffer(context, cl.mem_flags.READ_WRITE, h_un1.nbytes)
	d_udn1 = cl.Buffer(context, cl.mem_flags.READ_WRITE, h_udn1.nbytes)

	# Write only
	d_damage = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, h_damage.nbytes)
	# Initialize kernel parameters
	cl_kernel_initial_values.set_scalar_arg_dtypes([None, None])
	cl_kernel_time_marching_1.set_scalar_arg_dtypes([None, None, None, None, None])
	cl_kernel_time_marching_2.set_scalar_arg_dtypes([None, None, None, None, None])
	cl_kernel_time_marching_3.set_scalar_arg_dtypes([None, None, None, None])
	cl_kernel_check_bonds.set_scalar_arg_dtypes([None, None, None])
	cl_kernel_calculate_damage.set_scalar_arg_dtypes([None, None, None])
	
# =============================================================================
# 	# Covariance matrix
# 	K = myModel.K
# 	
# 	# Cholesky decomposition of K
# 	C = myModel.C
# =============================================================================
	
	global_size = int(myModel.DPN * myModel.nnodes)
	cl_kernel_initial_values(queue, (global_size,), None, d_un, d_udn)
	for t in range(1, numSteps):
		
		st = time.time()
		
		cl_kernel_time_marching_1(queue, (myModel.DPN * myModel.nnodes,), None, d_un1, d_un, d_udn, d_bctypes, d_bcvalues)
		cl_kernel_time_marching_2(queue, (myModel.nnodes,), None, d_un1, d_udn1, d_vols, d_horizons, d_coords)
		cl_kernel_time_marching_3(queue, (myModel.DPN * myModel.nnodes,), None, d_un, d_udn, d_un1, d_udn1)
		cl_kernel_check_bonds(queue, (myModel.nnodes, myModel.MAX_HORIZON_LENGTH), None, d_horizons, d_un, d_coords)
		
		if(t % print_every == 0):
			cl_kernel_calculate_damage(queue, (myModel.nnodes,), None, d_damage, d_horizons, d_horizons_lengths)
			cl.enqueue_copy(queue, h_damage, d_damage)
			cl.enqueue_copy(queue, h_un, d_un)
			print("Sum of all damage is", np.sum(h_damage))
			vtk.write("U_"+"t"+str(t)+".vtk","Solution time step = "+str(t), myModel.coords, h_damage, h_un)
			
		print('Timestep {} complete in {} s '.format(t, time.time() - st))
		
	# final time step
	cl_kernel_calculate_damage(queue, (myModel.nnodes,), None, d_damage, d_horizons, d_horizons_lengths)
	cl.enqueue_copy(queue, h_damage, d_damage)
	cl.enqueue_copy(queue, h_un, d_un)
	vtk.write("U_"+"t"+str(t)+".vtk","Solution time step = "+str(t), myModel.coords, h_damage, h_un)
	return vtk.write("U_"+"sample"+str(sample)+".vtk","Solution time step = "+str(t), myModel.coords, h_damage, h_un)




def main():
	""" Stochastic Peridynamics, takes multiple stable states (fully formed cracks)
	"""

	st =  time.time()	
	thisModel = simpleSquare()
	no_samples = 1
	for s in range(no_samples):
		sim(s,thisModel)
	print('TOTAL TIME REQUIRED {}'.format(time.time() - st))
main()

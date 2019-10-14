
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../../Peridynamics')
sys.path.insert(1, '../../PostProcessing')

import PeriParticle as peri
import ParPeri as sim
import numpy as np
import vtu as vtu

from mpi4py import MPI
from timeit import default_timer as timer

from numpy import linalg as LA



class param:

	# A User defined class for a particular problem which defines all necessary parameters

	def __init__(self):

		self.dim = 2

		self.meshFileName = 'test.msh'

		self.meshType = 2
		self.boundaryType = 1
		self.numBoundaryNodes = 2
		self.numMeshNodes = 3

		# Material Parameters from classical material model
		self.horizon = 0.1
		self.K = 0.05
		self.s00 = 0.05

		self.tabLength = 20.0
		self.taperLength = 10.0
		self.innerLength = 115.0

		self.totalWidth = 20.0

		self.notchWidth = 5.0
		self.notchDepth = 5.0

		self.totalLength = 2 * self.tabLength + 2 * self.taperLength + self.innerLength

		self.c = 18.0 * self.K / (np.pi * (self.horizon**4))


	def findBoundary(self,x):
		# Function which markes constrain particles
	    bnd = 0 # Does not live on a boundary
	    if (x[0] < (self.horizon)):
	        bnd = -1
	    elif (x[0] > (1.0 - self.horizon)):
	        bnd = 1
	    return bnd

	def isCrack(self, x, y):
		output = 0
		return output






## Start of main


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
comm_Size = comm.Get_size()

modelParameters = param()

myModel = sim.ParModel(modelParameters, comm)


# Starting Time loop

numSteps = 1000

sigma = 0.0

loadRate = 0.1

dt = 1e-3

print_every = 10

rank = comm.Get_rank()

verb = 0
if(rank == 0):
 	verb = 1

if(verb > 0):
	print("Peridynamic Simulation -- Starting")

u = []

damage = []

# Setup broken flag

broken = []
for i in range(0, myModel.numlocalNodes):
	broken.append(np.zeros(len(myModel.family[i])))


u.append(np.zeros((myModel.nnodes, 3)))

damage.append(np.zeros(myModel.numlocalNodes)) # no damage in the first step

d = []
d.append(np.zeros(myModel.nnodes))
for t in range(1, numSteps):
	d.append(np.zeros(myModel.nnodes))


time = 0.0;

	# Apply boundary conditions
for t in range(1, numSteps):

	time += dt;
	if(verb > 0):
		print("Time step = " + str(t) + ", Time = " + str(time))


	# Communicate Ghost particles to required processors
	u[t-1] = myModel.communicateGhostParticles(u[t-1])

	damage.append(np.zeros(myModel.numlocalNodes))

	broken, damage[t] = myModel.checkBonds(u[t-1], broken, damage[t-1])

	f = myModel.computebondForce(u[t-1], broken)

	# Simple Euler update of the Solution + Add the Stochastic Random Noise
	u.append(np.zeros((myModel.nnodes, 3)))

	u[t][myModel.l2g,:] = u[t-1][myModel.l2g,:] + dt * f +  np.random.normal(loc = 0.0, scale = sigma, size = (myModel.numlocalNodes, 3))


	

	# Apply boundary conditions

	u[t][myModel.lhs,1:3] = np.zeros((len(myModel.lhs),2))
	u[t][myModel.rhs,1:3] = np.zeros((len(myModel.rhs),2))
	u[t][myModel.lhs,0] = np.zeros(len(myModel.lhs))
	u[t][myModel.rhs,0] = time * loadRate * np.ones(len(myModel.rhs))


	# Write Solution to file every N steps
	if ((t % print_every == 0)):
		vtu.writeParallel("U_" + str(t),comm, myModel.numlocalNodes, myModel.coords[myModel.l2g,:], damage[t], u[t][myModel.l2g,:])

	comm.Barrier()

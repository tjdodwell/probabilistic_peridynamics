
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../../Peridynamics')
sys.path.insert(1, '../../PostProcessing')

import PeriParticle as peri
from SeqPeri import SeqModel as MODEL
import numpy as np
import vtk as vtk
from timeit import default_timer as timer


class simpleSquare(MODEL):

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
		self.s00 = 0.005

		self.crackLength = 0.3

		self.readMesh(self.meshFileName)

		self.setNetwork(self.horizon)

		self.lhs = []
		self.rhs = []

		# Find the Boundary
		for i in range(0, self.nnodes):
			bnd = self.findBoundary(self.coords[i][:])
			if (bnd < 0):
				(self.lhs).append(i)
			elif (bnd > 0):
				(self.rhs).append(i)

	def findBoundary(self,x):
		# Function which markes constrain particles
	    bnd = 0 # Does not live on a boundary
	    if (x[0] < 1.5 * self.horizon):
	        bnd = -1
	    elif (x[0] > 1.0 - 1.5 * self.horizon):
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

## Start of main

myModel = simpleSquare()

#myModel = sim.SeqModel(modelParameters)


# Time loop

numSteps = 1000

sigma = 1e-5

loadRate = 0.00001

dt = 1e-3

print_every = 10

print("Peridynamic Simulation -- Starting")

u = []

damage = []

# Setup broken flag

broken = []
for i in range(0, myModel.nnodes):
	broken.append(np.zeros(len(myModel.family[i])))

u.append(np.zeros((myModel.nnodes, 3)))

damage.append(np.zeros(myModel.nnodes))

broken, damage[0] = myModel.initialiseCrack(broken, damage[0])

verb = 1

time = 0.0;


for t in range(1, numSteps):

	time += dt;

	if(verb > 0):
		print("Time step = " + str(t) + ", Time = " + str(time))

	# Compute the force with displacement u[t-1]

	damage.append(np.zeros(myModel.nnodes))

	broken, damage[t] = myModel.checkBonds(u[t-1], broken, damage[t-1])

	f = myModel.computebondForce(u[t-1], broken) # myModel.computeForce(u[t-1])

	# Simple Euler update of the Solution + Add the Stochastic Random Noise

	u.append(np.zeros((myModel.nnodes, 3)))

	u[t] = u[t-1] + dt * f + np.random.normal(loc = 0.0, scale = sigma, size = (myModel.nnodes, 3))

	# Apply boundary conditions
	u[t][myModel.lhs,1:3] = np.zeros((len(myModel.lhs),2))
	u[t][myModel.rhs,1:3] = np.zeros((len(myModel.rhs),2))

	u[t][myModel.lhs,0] = -0.5 * t * loadRate * np.ones(len(myModel.rhs))
	u[t][myModel.rhs,0] = 0.5 * t * loadRate * np.ones(len(myModel.rhs))

	# Write Solution to file every N steps
	if (verb & (t % print_every == 0)):
		vtk.write("U_"+str(t)+".vtk","Solution time step = "+str(t), myModel.coords, damage[t], u[t])

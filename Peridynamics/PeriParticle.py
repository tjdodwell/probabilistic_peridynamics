class Particle:

	def __init__(self):
		self.family = []
		self.V = 0.0;
		self.U = []
		self.tsteps = 1
		self.damage = []
		self.damage.append(0)

	def setDisp(self, U_):
		self.U.append(U_)

	def getDisp(self, time):
		return self.U[time]

	def setDamage(self, val_, time):
		if (time > self.tsteps):
			self.damage.append(val_)
		else :
			self.damage[time] = val_

	def getDamage(self,time):
		return self.damage[time]

	def setId(self, id_):
		self.id = id_

	def setCoord(self, x_):
		self.x = x_

	def addNeighbour(self, j):
		(self.family).append(j)

	def getFamily(self):
		return self.family

	def getX(self):
		return self.x

	def addVolume(self, vol_):
		self.V += vol_

	def getVolume(self):
		return self.V

	def getFamilySize(self):
		return len(self.family)

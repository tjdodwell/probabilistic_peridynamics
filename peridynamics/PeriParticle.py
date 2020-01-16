class Particle:
    def __init__(self):
        self.family = []
        self.V = 0.0
        self.U = []
        self.tsteps = 1
        self.damage = []
        self.damage.append(0)

    def setDisp(self, U):
        self.U.append(U)

    def getDisp(self, time):
        return self.U[time]

    def setDamage(self, val, time):
        if (time > self.tsteps):
            self.damage.append(val)
        else:
            self.damage[time] = val

    def getDamage(self, time):
        return self.damage[time]

    def setId(self, _id):
        self.id = _id

    def setCoord(self, x):
        self.x = x

    def addNeighbour(self, j):
        self.family.append(j)

    def getFamily(self):
        return self.family

    def getX(self):
        return self.x

    def addVolume(self, vol):
        self.V += vol

    def getVolume(self):
        return self.V

    def getFamilySize(self):
        return len(self.family)

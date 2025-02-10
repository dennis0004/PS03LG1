
import numpy as np

class Agent2611gb: # Qlg1 (bidi-Q), nC = nW, greedy; Qps (standard Q) Boltzmann
    def __init__(self, nC, nAgent, neighbourList, lrLG, lrPS, tauPS, aID, name='', isDebug=False):
        self.name = name
        self.aID = aID
        self.nAgent = nAgent
        self.isDebug = isDebug

        self.lrLG = lrLG

        self.lrPS = lrPS
        self.tauPS = tauPS

        self.nC = nC
        self.Qc2w = np.zeros((nC, nC))
        self.Qw2c = np.zeros((nC, nC))

        self.neighbourList = neighbourList
        self.Qps = np.zeros(len(neighbourList))  # Q-values for neighbours

    # ------------------------------------------------------------------
    def getQps(self, s):
        return self.Qps

    def getPolicyPS(self, s):
        return self.softmax(self.tauPS * self.getQps(0))

    def getQps1(self, s):
        Qps1 = np.zeros(self.nAgent)
        for a, q in enumerate(self.Qps):
            Qps1[self.neighbourList[a]] = q
        return Qps1

    def getPolicyPS1(self, s):
        Pps1 = np.zeros(self.nAgent)
        Pps = self.getPolicyPS(0)
        for a, p in enumerate(Pps):
            Pps1[self.neighbourList[a]] = p
        return Pps1

    def getActionPS(self, s): # Boltzmann, return opponent ID
        policyPsS = self.getPolicyPS(0)
        action = np.random.choice(len(policyPsS), p=policyPsS)
        opID = self.neighbourList[action]
        return opID

    def train1ps(self, s, opID, r):
        a = self.neighbourList.index(opID)
        self.Qps[a] = self.Qps[a] + self.lrPS * (r - self.Qps[a])

    def softmax(self, x):
        z = x - np.max(x)
        numerator = np.exp(z)
        denominator = np.sum(numerator)
        softmax = numerator / denominator
        return softmax

    # ------------------------------------------------------------------
    def getQlgAll(self):
        Qlg = np.concatenate((self.Qc2w, self.Qw2c), axis=0)
        return np.ravel(Qlg)

    def getQlg(self, s): # c2w: 0-99, w2c: 100-199
        if s < 100:
            return self.Qc2w[s]
        else:
            return self.Qw2c[s - 100]

    def getPlgAll(self):
        Qlg = np.concatenate((self.Qc2w, self.Qw2c), axis=0)
        Plg = Qlg == np.amax(Qlg, axis=1, keepdims=True)
        Plg = Plg / np.sum(Plg, axis=1, keepdims=True)
        return np.ravel(Plg)

    def getPlg(self, s):
        if s < 100: # c to w
            Qs = self.Qc2w[s]
        else: # w to c
            Qs = self.Qw2c[s - 100]
        piS = Qs == np.amax(Qs)
        piS = piS / np.sum(piS)
        return piS

    def getC2WArray(self):
        c2wA = np.ones(self.nC, dtype=int) * -1

        for c in range(self.nC):
            Qs = self.Qc2w[c]
            aL = np.argwhere(Qs == np.amax(Qs)).ravel()
            w = np.random.choice(aL)
            c2wA[c] = w
        return c2wA

    def getW2CArray(self):
        w2cA = np.ones(self.nC, dtype=int) * -1

        for w in range(self.nC):
            Qa = self.Qw2c[w]
            sL = np.argwhere(Qa == np.amax(Qa)).ravel()
            c = np.random.choice(sL)
            w2cA[w] = c
        return w2cA

    def getActionLG(self, s):  # greedy
        if s < 100:  # c to w
            Qs = self.Qc2w[s]
        else:  # w to c
            Qs = self.Qw2c[s - 100]

        aA = np.argwhere(Qs == np.amax(Qs)).ravel()
        a = np.random.choice(aA)
        return a

    def train1lg(self, s, a, r):
        if s < 100:
            self.Qc2w[s, a] = self.Qc2w[s, a] + self.lrLG * (r - self.Qc2w[s, a])
        else:
            s = s - 100
            self.Qw2c[s, a] = self.Qw2c[s, a] + self.lrLG * (r - self.Qw2c[s, a])

if __name__ == '__main__':
    agent = Agent2611gb(10, 100, [0,1,2,3,4], 0.95, 0.1, -10, 0)
    print(agent.getActionLG(0))





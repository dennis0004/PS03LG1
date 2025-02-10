
import numpy as np

class LangGame11:
    def __init__(self, Nconcept):
        self.Nconcept = Nconcept

    def reset(self):
        self.state = np.random.choice(self.Nconcept)
        return self.state

    def step(self, action):
        if self.state == action:
            reward = 1
        else:
            reward = -1
        self.state = np.random.choice(self.Nconcept)
        return self.state, reward

if __name__ == '__main__':
    game = LangGame11(10)
    s1 = game.reset()
    s1_, r = game.step(s1)
    print(r)

    s1 = s1_
    s1_, r = game.step(0)
    print(r)




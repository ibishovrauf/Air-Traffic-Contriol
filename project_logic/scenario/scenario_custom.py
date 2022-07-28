import bluesky as bs

class Scenario:
    def __init__(self, ac_n = 5):
        self.ac_n = ac_n
    def create_scenario(self):
        bs.traf.mcre(n=self.ac_n)

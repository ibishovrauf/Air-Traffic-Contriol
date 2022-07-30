from bluesky import traf
class Scenario:
    def __init__(self, ac_n = 5):
        self.traf = traf()

    def create_scenario(self):
        self.traf.mcre(n=5)
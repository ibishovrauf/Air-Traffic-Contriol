from bluesky import traf
from bluesky import s
class Scenario:
    def __init__(self, ac_n = 5):
        self.traf = traf()

    def create_scenario(self):
        self.traf.mcre(n=5)

#!/usr/bin/env python
""" Pygame BlueSky start script """
from __future__ import print_function
import pygame as pg
import bluesky as bs
from bluesky.ui.pygame import splash
from dataclasses import dataclass
import time
import numpy as np


@dataclass
class AirCraft:
    id: int
    lat: float
    lon: float
    alt: float
    heading: float
    speed: float
    aceleration: float

    def distance(self, aircraft):
        return np.sqrt((self.alt - aircraft.alt) ** 2 + ((self.lat - aircraft.lat) * 111.139) ** 2 + (
                    (self.lon - aircraft.lon) * 111.139) ** 2)

    def is_instate(self, aircraft):
        if (abs(self.lon - aircraft.lon)) * 111.139 < 300 and (abs(self.lat - aircraft.lat)) * 111.139 < 300 and (
        abs(self.alt - aircraft.alt)) < 1000:
            return aircraft
        return None

    def state(self, aircraft):
        return [(self.alt - aircraft.alt) // 300, (self.lat - aircraft.lat) * 111.139 // 20,
                (self.lon - aircraft.lon) * 111.139 // 20]


def main():
    """ Start the mainloop (and possible other threads) """
    splash.show()
    bs.init(gui='pygame')
    # bs.sim.op()
    bs.scr.init()

    # Main loop for BlueSky
    while not bs.sim.state == bs.END:
        bs.sim.step()  # Update sim
        bs.scr.update()  # GUI update

        if bs.traf.cd.confpairs:
            for conflict in bs.traf.cd.confpairs_unique:
                index_2 = bs.traf.id.index(list(conflict)[1])
                index_1 = bs.traf.id.index(list(conflict)[0])
                bs.traf.alt[index_1] -= 50
                bs.traf.alt[index_2] += 50
        if bs.traf.ntraf > 0:
            get_state(bs.traf.id[0], bs.traf)
    bs.sim.quit()
    pg.quit()

    print('BlueSky normal end.')


def get_state(current: int, traffic):
    aircrafts = []
    for index in range(len(traffic.id)):
        if traffic.id[index] == current:
            current = index
        aircraft = AirCraft(
            id=traffic.id[index],
            aceleration=traffic.ax[index],
            lat=traffic.lat[index],
            lon=traffic.lon[index],
            alt=traffic.alt[index],
            heading=traffic.hdg[index],
            speed=traffic.tas[index]
        )
        aircrafts.append(aircraft)
    current = aircrafts.pop(current)
    listik = list()
    for aircraft in aircrafts:
        if current.is_instate(aircraft) is not None:
            listik.append(aircraft)
            print(current.state(aircraft))


if __name__ == '__main__':
    print("   *****   BlueSky Open ATM simulator *****")
    print("Distributed under GNU General Public License v3")
    # Run mainloop if BlueSky_pygame is called directly
    main()
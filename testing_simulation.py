""" Pygame BlueSky start script """
from __future__ import print_function
import numpy as np
import random
import timeit
import pygame as pg
import bluesky as bs
from bluesky.ui.pygame import splash

from project_logic.utils import save_testing_parameters
import timeit
import numpy as np
from dataclasses import dataclass
import random
from collections import defaultdict

import copy


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

    def is_in_state(self, aircraft):
        if (abs(self.lon - aircraft.lon)) * 111.139 < 100_000 and (
        abs(self.lat - aircraft.lat)) * 111.139 < 100_000 and (abs(self.alt - aircraft.alt)) < 3_000:
            return True
        return False

    def state_index(self, aircraft):
        return (int((self.alt - aircraft.alt) // 300), int((self.lat - aircraft.lat) * 111.139 // 10_000),
                int((self.lon - aircraft.lon) * 111.139 // 10_000))

    def __eq__(self, aircraft) -> bool:
        if not isinstance(aircraft, str):
            return self.id == aircraft.id
        return self.id == aircraft  # if aircraft is string that contein only id


class Simulation:
    def __init__(self, Model, n_traf, max_steps, state_shape, num_actions, plot_path):
        self._Model = Model
        self._n_traf = n_traf
        self._state_shape = state_shape
        self._num_actions = num_actions
        self._step = 0
        self._plot_path = plot_path
        self._max_steps = max_steps
        self._aircrafts = np.array([])
        self._rewards = np.array([])
        self._AltCmd = 0
        self._SpdCmd = 0
        self._action_dict = defaultdict(list)

        bs.init(gui='pygame')
        self._AirTraffic = bs.traf
        self._confpairs_dict = dict()
        self._actions = np.array([])

    def run(self) -> list:
        """
        Runs the testing simulation
        """
        start_time = timeit.default_timer()

        splash.show()
        bs.scr.init()

        # inits
        self._step = 0
        self._generate_conf_aircraft(self._n_traf, self._n_traf)

        while not bs.sim.state == bs.END and self._step < self._max_steps:

            self._step += 1
            bs.sim.step()  # Update sim
            bs.scr.update()  # GUI update

            if bs.traf.cd.confpairs:
                self._create_aircraft_list()
                for aircrafts in self._AirTraffic.cd.confpairs_unique:
                    aircrafts = list(aircrafts)
                    if "_".join(aircrafts) in self._confpairs_dict.keys():
                        self._confpairs_dict["_".join(aircrafts)] += 1
                    else:
                        self._confpairs_dict["_".join(aircrafts)] = 1
                    for aircraft in aircrafts:
                        current_state = self._get_state(aircraft)

                        reward = 0

                        action = self._choose_action(current_state)

                        self._set_action(action, aircraft)
                        self._actions = np.append(self._actions, action)

                        old_state = current_state
                        old_action = action

                        bs.sim.step()
                        bs.scr.update()

                        current_state = self._get_state(aircraft)
                        reward = self._calc_reward(action, aircraft, start_time, self._step)
                        self._rewards = np.append(self._rewards, reward)
        bs.sim.quit()
        pg.quit()
        save_testing_parameters(self._rewards, self._actions, self._confpairs_dict, self._plot_path)
        simulation_time = round(timeit.default_timer() - start_time, 1)
        return simulation_time

    def _generate_conf_aircraft(self, n_norm_ac: int, n_conf_ac: int):
        self._AirTraffic.cd.setmethod(name='ON')
        self._AirTraffic.mcre(n_conf_ac, acalt=12000, acspd=100)
        for index in range(len(self._AirTraffic.id)):
            dpsi = np.random.choice([30, 60, 90], 1)
            tlosh = np.random.randint(low=400, high=600)
            idtmp = chr(np.random.randint(65, 90)) + chr(np.random.randint(65, 90)) + '{:>05}'
            acid = idtmp.format(index)
            self._AirTraffic.creconfs(acid=acid, actype='B744', targetidx=index, dpsi=dpsi, dcpa=2.5, tlosh=tlosh)

        self._AirTraffic.mcre(n_norm_ac, acalt=10000, acspd=100)
        self._init_aircrafts = copy.copy(self._AirTraffic)

    def _get_state(self, current_aircraft):
        state = np.zeros(self._state_shape)
        for i in self._aircrafts:
            if i == current_aircraft:
                current_aircraft = i

        for aircraft in self._aircrafts:
            if current_aircraft.is_in_state(aircraft):
                aircraft_values = np.array([aircraft.lat,
                                            aircraft.lon,
                                            aircraft.alt,
                                            aircraft.heading,
                                            aircraft.speed,
                                            aircraft.aceleration])

                state[current_aircraft.state_index(aircraft)] = aircraft_values
        return state.reshape(1, -1)[0]

    def _create_aircraft_list(self):
        for index in range(len(self._AirTraffic.id)):
            aircraft = AirCraft(
                id=self._AirTraffic.id[index],
                aceleration=self._AirTraffic.ax[index],
                lat=self._AirTraffic.lat[index],
                lon=self._AirTraffic.lon[index],
                alt=self._AirTraffic.alt[index],
                heading=self._AirTraffic.hdg[index],
                speed=self._AirTraffic.tas[index]
            )
            self._aircrafts = np.append(self._aircrafts, aircraft)

    def _choose_action(self, state):
        return np.argmax(self._Model.predict_one(state))  # the best action given the current state

    def _set_action(self, action, aircraft):
        aircraft_index = self._AirTraffic.id.index(aircraft)
        if action == 0:
            self._AirTraffic.alt[aircraft_index] -= 1200
            self._AirTraffic.selalt[aircraft_index] -= 1200
            self._AltCmd = -1200
        elif action == 1:
            self._AirTraffic.alt[aircraft_index] -= 600
            self._AirTraffic.selalt[aircraft_index] -= 600
            self._AltCmd = -600
        elif action == 2:
            self._AirTraffic.alt[aircraft_index] += 600
            self._AirTraffic.selalt[aircraft_index] += 600
            self._AltCmd = 600
        elif action == 3:
            self._AirTraffic.alt[aircraft_index] += 1200
            self._AirTraffic.selalt[aircraft_index] += 1200
            self._AltCmd = 1200
        elif action == 4:
            self._AirTraffic.hdg[aircraft_index] += 6
            self._AirTraffic.ap.trk[aircraft_index] += 6
        elif action == 5:
            self._AirTraffic.hdg[aircraft_index] -= 6
            self._AirTraffic.ap.trk[aircraft_index] -= 6
        elif action == 6:
            self._AirTraffic.tas[aircraft_index] += 30
            self._AirTraffic.selspd[aircraft_index] += 30
            self._SpdCmd = 30
        elif action == 7:
            self._AirTraffic.tas[aircraft_index] += 20
            self._AirTraffic.selspd[aircraft_index] += 20
            self._SpdCmd = 20
        elif action == 8:
            self._AirTraffic.tas[aircraft_index] += 10
            self._AirTraffic.selspd[aircraft_index] += 10
            self._SpdCmd = 10
        elif action == 9:
            self._AirTraffic.tas[aircraft_index] -= 10
            self._AirTraffic.selspd[aircraft_index] -= 10
            self._SpdCmd = -10
        elif action == 10:
            self._AirTraffic.tas[aircraft_index] -= 20
            self._AirTraffic.selspd[aircraft_index] -= 20
            self._SpdCmd = -20
        elif action == 11:
            self._AirTraffic.tas[aircraft_index] -= 30
            self._AirTraffic.selspd[aircraft_index] -= 30
            self._SpdCmd = -30
        else:
            return None

    def _calc_reward(self, action, ac_id, start_time, now_time):
        """
        Reward can be divided into infeasible solution and feasible solution.
        The infeasible solution refers to the solution beyond the scope of aircraft
        performance or in violation of actual control habits, such as the climbing action
        followed by the descending action. The feasible solution is the solution other
        than the infeasible solution.
        """
        self._action_dict[ac_id].append(action)
        if len(self._action_dict[ac_id]) > 1:
            if (self._action_dict[ac_id][-1] in [0, 1] and self._action_dict[ac_id][-2] in [2, 3]) \
                    or (self._action_dict[ac_id][-1] in [2, 3] and self._action_dict[ac_id][-2] in [0, 1]):
                return -1
        aircraft_index = self._AirTraffic.id.index(ac_id)
        init_hdg = self._init_aircrafts.hdg[aircraft_index]
        after_hdg = self._AirTraffic.hdg[aircraft_index]
        if init_hdg < 0 and abs(after_hdg - init_hdg)>180:
            init_hdg = 360 - abs(init_hdg)
        elif abs(after_hdg - init_hdg) > 180:
            init_hdg = 360 - abs(init_hdg)
        r_a = 1 - abs(self._init_aircrafts.alt[aircraft_index] - self._AirTraffic.alt[aircraft_index])/2000
        r_s = 0.95 - abs(self._init_aircrafts.tas[aircraft_index] - self._AirTraffic.tas[aircraft_index])/100
        r_h = 0.001*abs(init_hdg - after_hdg)
        r_idv = r_a + r_s + r_h

        # r_a = 1 - abs(self._AltCmd) / 2000
        # r_s = 0.95 - abs(self._SpdCmd) / 100
        # r_h = 0.3  #
        # r_idv = r_a + r_h + r_s
        bs.sim.step()
        conf_aircrafts = []

        r_overall = 0
        if bs.traf.cd.confpairs:
             for aircrafts in self._AirTraffic.cd.confpairs_unique:
                 aircrafts = list(aircrafts)
                 conf_aircrafts.extend(aircrafts)

        if conf_aircrafts.count(ac_id) >= 2:
            r_overall = -3
        elif conf_aircrafts.count(ac_id) == 1:
            r_overall = -0.6
        elif ac_id not in conf_aircrafts:
            r_overall = 1 - (timeit.default_timer() - start_time) / 180

        return r_idv + r_overall

    @property
    def num_states(self):
        return self._Model.input_dim
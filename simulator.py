""" Pygame BlueSky start script """
from __future__ import print_function
import pygame as pg
import bluesky as bs
from bluesky.ui.pygame import splash


from project_logic.utils import remember_rewards
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
        return np.sqrt((self.alt - aircraft.alt)**2 + ((self.lat - aircraft.lat)*111.139)**2 + ((self.lon - aircraft.lon)*111.139)**2)

    def is_in_state(self, aircraft):
        if (abs(self.lon - aircraft.lon))*111.139 < 100_000 and (abs(self.lat - aircraft.lat))*111.139 < 100_000 and (abs(self.alt - aircraft.alt)) < 3_000:
            return True
        return False

    def state_index(self, aircraft):
        return (int((self.alt - aircraft.alt)//300), int((self.lat - aircraft.lat)*111.139//10_000) ,int((self.lon - aircraft.lon)*111.139//10_000))

    def __eq__(self, aircraft) -> bool:
        if not isinstance(aircraft, str):
            return self.id == aircraft.id
        return self.id == aircraft # if aircraft is string that contein only id


class Simulator:

    def __init__(self, memory, model, gamma, n_traf, state_shape, num_actions, max_step, training_epochs) -> None:
        self._Memory = memory
        self._Model = model
        self._gamma = gamma
        self._n_traf = n_traf
        self._state_shape = state_shape
        self._num_actions = num_actions
        self._step = 0
        self._max_step = max_step
        self._training_epochs = training_epochs
        self._aircrafts = np.array([])
        self._init_aircrafts = object
        self._rewards = np.array([])
        self._AltCmd = 0
        self._SpdCmd = 0
        self._action_dict = defaultdict(list)

        bs.init(gui='pygame')
        self._AirTraffic = bs.traf


    def run(self, episode, epsilon) -> list:
        """
        Runs an episode of simulation, then starts a training session
        """
        self._aircrafts = np.array([])
        self._rewards = np.array([])
        self._AltCmd = 0
        self._SpdCmd = 0

        start_time = timeit.default_timer()

        # first, generate the route file for this simulation

        splash.show()
        # bs.sim.op()
        bs.scr.init()

        step=0
        self._generate_conf_aircraft(self._n_traf, self._n_traf)
        while not bs.sim.state == bs.END and step < self._max_step:
            step+=1
            bs.sim.step() # Update sim
            bs.scr.update()   # GUI update
            if bs.traf.cd.confpairs:
                self._create_aircraft_list()
                for aircrafts in self._AirTraffic.cd.confpairs_unique:
                    aircrafts = list(aircrafts)
                    for aircraft in aircrafts:
                        current_state = self._get_state(aircraft)
                        reward = 0

                        action = self._choose_action(current_state, epsilon)

                        self._set_action(action, aircraft)

                        old_state = current_state
                        old_action = action

                        bs.sim.step()
                        bs.scr.update()

                        current_state = self._get_state(aircraft)
                        reward = self._calc_reward(action, aircraft,start_time)
                        self._rewards = np.append(self._rewards, reward)
                        self._Memory.add_sample((old_state, old_action, reward, current_state))


        text = "Total reward: " + str(round(self._rewards.mean(), 3)) + " - Epsilon:" + str(round(epsilon, 2))+"\n"
        remember_rewards(self._rewards, text, epsilon)
        bs.sim.quit()
        pg.quit()
        self._AirTraffic.reset()
        simulation_time = round(timeit.default_timer() - start_time, 1)

        print("Training...")
        start_time = timeit.default_timer()
        for _ in range(self._training_epochs):
            self._replay()
        training_time = round(timeit.default_timer() - start_time, 1)

        return simulation_time, training_time

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

    def _get_state(self, current_aircraft):
        state = np.zeros(self._state_shape)
        for i in self._aircrafts:
            if i == current_aircraft:
                current_aircraft = i

        for aircraft in self._aircrafts:
            if current_aircraft.is_in_state(aircraft):
                aircraft_values = np.array([    aircraft.lat,
                                                aircraft.lon,
                                                aircraft.alt,
                                                aircraft.heading,
                                                aircraft.speed,
                                                aircraft.aceleration])

                state[current_aircraft.state_index(aircraft)] = aircraft_values
        return state.reshape(1, -1)[0]

    def _generate_conf_aircraft(self, n_norm_ac: int, n_conf_ac: int):
        self._AirTraffic.cd.setmethod(name='ON')
        self._AirTraffic.mcre(n_conf_ac, acalt=12000, acspd=100)
        for index in range(len(self._AirTraffic.id)):
            # target.append(self._AirTraffic.id[index])
            dpsi = np.random.choice([30,60,90],1)
            tlosh = np.random.randint(low=400, high=600)
            idtmp = chr(np.random.randint(65, 90)) + chr(np.random.randint(65, 90)) + '{:>05}'
            acid = idtmp.format(index)
            self._AirTraffic.creconfs(acid=acid, actype='B744', targetidx=index, dpsi=dpsi, dcpa=2.5, tlosh=tlosh)

        self._AirTraffic.mcre(n_norm_ac, acalt=10000, acspd=100)
        self._init_aircrafts = copy.copy(self._AirTraffic)

    def _choose_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randint(0, self._num_actions - 1) # random action
        else:
            return np.argmax(self._Model.predict_one(state)) # the best action given the current state

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
            self._AirTraffic.selspd[aircraft_index] +=30
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


    def _calc_reward(self, action, ac_id, start_time):
        """
        Estimate reward of chosen action

        Params:
            action: [0-11].
            ac_id: id of conflicting aircraft
            start_time: when the simulation started
        """

        # Infeasible solution. If our aircraft chose the climbing action(positive altitude) followed by the
        # descending action(negative altitude)
        self._action_dict[ac_id].append(action)
        if len(self._action_dict[ac_id]) > 1:
            if (self._action_dict[ac_id][-1] in [0, 1] and self._action_dict[ac_id][-2] in [2, 3]) \
                    or (self._action_dict[ac_id][-1] in [2, 3] and self._action_dict[ac_id][-2] in [0, 1]):
                return -1

        # Feasible solution
        aircraft_index = self._AirTraffic.id.index(ac_id)
        init_hdg = self._init_aircrafts.hdg[aircraft_index]
        after_hdg = self._AirTraffic.hdg[aircraft_index]
        if init_hdg < 0 and abs(after_hdg - init_hdg) > 180:
            init_hdg = 360 - abs(init_hdg)
        elif abs(after_hdg - init_hdg) > 180:
            init_hdg = 360 - abs(init_hdg)
        r_a = 1 - abs(self._init_aircrafts.alt[aircraft_index] - self._AirTraffic.alt[aircraft_index]) / 600
        r_s = 1 - abs(self._init_aircrafts.tas[aircraft_index] - self._AirTraffic.tas[aircraft_index]) / 10
        r_h = 1 - abs(init_hdg - after_hdg) / 6
        r_idv = r_a + r_s + r_h

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

    def _replay(self):

        batch = self._Memory.get_samples(self._Model.batch_size)
        if len(batch) > 0:  # if the memory is full enough
            states = np.array([val[0] for val in batch])  # extract states from the batch
            next_states = np.array([val[3] for val in batch])  # extract next states from the batch

            q_s_a = self._Model.predict_batch(states)  # predict Q(state), for every sample
            q_s_a_d = self._Model.predict_batch(next_states)  # predict Q(next_state), for every sample

            # setup training arrays
            x = np.zeros((len(batch), self.num_states))
            y = np.zeros((len(batch), self._num_actions))

            for i, b in enumerate(batch):
                state, action, reward, _ = b[0], b[1], b[2], b[3]  # extract data from one sample
                current_q = q_s_a[i]  # get the Q(state) predicted before
                current_q[action] = reward + self._gamma * np.amax(q_s_a_d[i])  # update Q(state, action)
                x[i] = state
                y[i] = current_q  # Q(state) that includes the updated action value

            self._Model.train_batch(x, y)  # train the NN

    @property
    def num_states(self):
        return self._Model.input_dim


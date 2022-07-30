""" Pygame BlueSky start script """
from __future__ import print_function
import pygame as pg
import bluesky as bs
from bluesky.ui.pygame import splash
import timeit
import numpy as np
from dataclasses import dataclass

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
        if (abs(self.lon - aircraft.lon))*111.139 < 300 and (abs(self.lat - aircraft.lat))*111.139 < 300 and (abs(self.alt - aircraft.alt)) < 1000:
            return True
        return False
    
    def state_index(self, aircraft):
        return [(self.alt - aircraft.alt)//300, (self.lat - aircraft.lat)*111.139//20 ,(self.lon - aircraft.lon)*111.139//20]

    def __eq__(self, aircraft) -> bool:
        if aircraft == object:
            return self.id == aircraft.id
        return self.id == aircraft # if aircraft is string that contein only id

class Simulator:

    def __init__(self, memory, model, gamma, n_simulation, state_shape, num_actions, max_step, training_epochs) -> None:
        self._Memory = memory
        self._Model = model
        self._gamma = gamma
        self._n_simulation = n_simulation
        self._state_shape = state_shape
        self._num_actions = num_actions 
        self._step = 0
        self._max_step = max_step
        self._training_epochs = training_epochs
        self._aircrafts = np.array([])

        bs.init(gui='pygame')
        self._AirTraffic = bs.traf

    def run(self, episode, epsilon) -> list:
        """
        Runs an episode of simulation, then starts a training session
        """
        start_time = timeit.default_timer()

        # first, generate the route file for this simulation
        # self._TrafficGen.genereta_scn(seed=episode)

        splash.show()
        # bs.sim.op()
        bs.scr.init()

        # Main loop for BlueSky
        step=0
        while not bs.sim.state == bs.END and step < 6000:
            step+=1

            bs.sim.step()   # Update sim
            bs.scr.update()   # GUI update
            if bs.traf.cd.confpairs:
                self._create_aircraft_list()
                for aircraft in self._AirTraffic.cd.confpairs_unique:
                    current_state = self._get_state(aircraft)
                    reward = 0
    
                    action = self._choose_action(current_state, epsilon)
                    self._set_action(action)
        
                    old_state = current_state
                    old_action = action

                    bs.sim.step()
                    bs.scr.update()

                    current_state = self._get_state(aircraft)
                    reward = self._calc_reward(action)
                    self._Memory.add_sample((old_state, old_action, reward, current_state))
   
        print("Total reward:", self._sum_neg_reward, "- Epsilon:", round(epsilon, 2))
        bs.sim.quit()
        pg.quit()
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
            np.add(self._aircrafts, aircraft)
        

    def _get_state(self, current_aircraft):
        state = np.zeros(self._state_shape)

        for aircraft in self._aircrafts:
            if current_aircraft.is_in_state(aircraft):
                aircraft_values = np.array([ aircraft.lat,
                                    aircraft.lon,
                                    aircraft.alt,
                                    aircraft.heading,
                                    aircraft.speed,
                                    aircraft.aceleration])
                state[current_aircraft.state_index(aircraft)] = aircraft

        return state

    def _choose_action(self, state, epsilon):
        pass

    def _set_action(self, action):
        pass

    def _calc_reward(self, action):
        pass

    def _replay(self):

        batch = self._Memory.get_samples(self._Model.batch_size)
        if len(batch) > 0:  # if the memory is full enough
            states = np.array([val[0] for val in batch])  # extract states from the batch
            next_states = np.array([val[3] for val in batch])  # extract next states from the batch

            q_s_a = self._Model.predict_batch(states)  # predict Q(state), for every sample
            q_s_a_d = self._Model.predict_batch(next_states)  # predict Q(next_state), for every sample

            # setup training arrays
            x = np.zeros((len(batch), self._num_states))
            y = np.zeros((len(batch), self._num_actions))

            for i, b in enumerate(batch):
                state, action, reward, _ = b[0], b[1], b[2], b[3]  # extract data from one sample
                current_q = q_s_a[i]  # get the Q(state) predicted before
                current_q[action] = reward + self._gamma * np.amax(q_s_a_d[i])  # update Q(state, action)
                x[i] = state
                y[i] = current_q  # Q(state) that includes the updated action value

            self._Model.train_batch(x, y)  # train the NN

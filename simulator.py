""" Pygame BlueSky start script """
from __future__ import print_function
import pygame as pg
import bluesky as bs
from bluesky.ui.pygame import splash

class Simulator:

    def __init__(self, memory, model, AirTraffic, gamma, n_simulation, state_shape, num_actions, max_step, training_epochs) -> None:
        self._memory = memory
        self._model = model
        self._AirTraffic = AirTraffic
        self._gamma = gamma
        self._n_simulation = n_simulation
        self._state_shape = state_shape
        self._num_actions = num_actions 
        self._step = 0
        self._max_step = max_step
        self._training_epochs = training_epochs

    def run(self, episode, epsilon) -> list:
        """
        Runs an episode of simulation, then starts a training session
        """
        splash.show()
        bs.init(gui='pygame')
        # bs.sim.op()
        bs.scr.init()

        # Main loop for BlueSky
        step=0
        while not bs.sim.state == bs.END and step < 6000:
            print(step)
            step+=1
            bs.sim.step()   # Update sim
            bs.scr.update()   # GUI update
        bs.sim.quit()
        pg.quit()

        print('BlueSky normal end.')


from dataclasses import dataclass


@dataclass
class AirCraft:
    lat: float
    long: float
    alt: float
    heading: float
    speed: float
    aceleration: float


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
        pass

import configparser
import datetime

from project_logic.Model import TestingModel
from project_logic.utils import set_test_path
from testing_simulation import Simulation

if __name__ == '__main__':
    content = configparser.ConfigParser()
    content.read('project_logic\\testing.ini')

    num_state = content['agent'].get('num_states')
    num_action = content['agent'].getint('num_actions')
    num_state = (eval(num_state))

    max_steps = content['simulation'].getint('max_steps')
    n_traf = content['simulation'].getint('n_traf')

    input_dim = content['model'].getint('input_dim')
    path = content['model'].get('path')
    model_number = content['model'].get('model_number')

    model_path, plot_path = set_test_path(path, model_number)

    model = TestingModel(
        input_dim=input_dim,
        model_path=model_path
    )
    simulation = Simulation(
        Model=model,
        n_traf=n_traf,
        state_shape=num_state,
        num_actions=num_action,
        max_steps=max_steps,
        plot_path=plot_path)

    episode = 0
    timestamp_start = datetime.datetime.now()

    print('\n----- Test episode')
    simulation_time = simulation.run()  # run the simulation
    print('Simulation time:', simulation_time, 's')



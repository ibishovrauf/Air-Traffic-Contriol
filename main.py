import configparser
import datetime

from project_logic.Memory import Memory
from project_logic.Model import Model
from project_logic.utils import set_train_path
from simulator import Simulator

if __name__ == '__main__':
    content = configparser.ConfigParser()
    content.read('project_logic\\training.ini')

    memory_min = content['memory'].getint('min_size')
    memory_max = content['memory'].getint('max_size')

    gamma = float(content['agent'].get('gamma'))
    num_state = content['agent'].get('num_states')
    num_action = content['agent'].getint('num_actions')
    num_state = (eval(num_state))

    max_steps = content['simulation'].getint('max_steps')
    total_episodes = content['simulation'].getint('total_episodes')
    n_traf = content['simulation'].getint('n_traf')

    batch_size = content['model'].getint('batch_size')
    num_layers = content['model'].getint('num_layers')
    input_dim = content['model'].getint('input_dim')
    width_layers = content['model'].getint('width_layers')
    learning_rate = content['model'].get('learning_rate')
    learning_rate = float(learning_rate)
    training_epochs = content['model'].getint('training_epochs')

    memory = Memory(max_size=memory_max, min_size=memory_min)
    model = Model(
        num_layers=num_layers,
        width=width_layers,
        batch_size=batch_size,
        learning_rate=learning_rate,
        input_dim=input_dim,
        output_dim=num_action
    )
    simulation = Simulator(
        memory=memory,
        model=model,
        gamma=gamma,
        n_traf=n_traf,
        state_shape=num_state,
        num_actions=num_action,
        max_step=max_steps,
        training_epochs=training_epochs)

    episode = 0
    timestamp_start = datetime.datetime.now()
    path = set_train_path()

    while episode < total_episodes:
        print('\n----- Episode', str(episode + 1), 'of', str(total_episodes))
        epsilon = 1.0 - (
                    episode / total_episodes)  # set the epsilon for this episode according to epsilon-greedy policy
        simulation_time, training_time = simulation.run(episode, epsilon)  # run the simulation
        print('Simulation time:', simulation_time, 's - Training time:', training_time, 's - Total:',
              round(simulation_time + training_time, 1), 's')
        episode += 1
        simulation._Model.save_model(path, episode)

    print("\n----- Start time:", timestamp_start)
    print("----- End time:", datetime.datetime.now())
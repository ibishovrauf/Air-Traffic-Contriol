import configparser
import datetime

from project_logic.Memory import Memory
from simulator import Simulator

if __name__ == '__main__':
    content = configparser.ConfigParser()
    content.read('project_logic\\training.ini')
    
    memory_min = content['memory'].getint('min_size')
    memory_max = content['memory'].getint('max_size')

    gamma = content['agent'].get('gamma')
    num_state = content['agent'].getint('num_state')
    num_action = content['agent'].getint('num_action')

    max_steps = content['simulation'].getint('max_steps')
    total_episodes = content['simulation'].getint('total_episodes')

    batch_size = content['model'].getint('batch_size')
    learning_rate = content['model'].getint('learning_rate')
    training_epochs = content['model'].getint('training_epochs')

    memory = Memory(max_size=memory_max, min_size=memory_min)
    model = None
    simulation = Simulator(
        memory=memory,
        model=model,
        gamma=gamma,
        n_simulation=None,
        state_shape=num_state,
        num_actions=num_action,
        max_step=max_steps,
        training_epochs=training_epochs)

    episode = 0
    timestamp_start = datetime.datetime.now()

    while episode < total_episodes:
        print('\n----- Episode', str(episode+1), 'of', str(total_episodes))
        epsilon = 1.0 - (episode / total_episodes)  # set the epsilon for this episode according to epsilon-greedy policy
        simulation_time, training_time = simulation.run(episode, epsilon)  # run the simulation
        print('Simulation time:', simulation_time, 's - Training time:', training_time, 's - Total:', round(simulation_time+training_time, 1), 's')
        episode += 1

    print("\n----- Start time:", timestamp_start)
    print("----- End time:", datetime.datetime.now())

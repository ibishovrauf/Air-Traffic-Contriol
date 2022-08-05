import os
import matplotlib.pyplot as plt

def set_train_path(models_path_name = 'model'):
    """
    Create a new model path with an incremental integer, also considering previously created model paths
    """
    models_path = os.path.join(os.getcwd(), models_path_name, '')
    os.makedirs(os.path.dirname(models_path), exist_ok=True)

    dir_content = os.listdir(models_path)
    if dir_content:
        previous_versions = [int(name.split("_")[1]) for name in dir_content]
        new_version = str(max(previous_versions) + 1)
    else:
        new_version = '1'

    data_path = os.path.join(models_path, 'model_'+new_version, '')
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    return data_path 

def data_path(data_path_name = 'foto'):
    data_path = os.path.join(os.getcwd(), data_path_name, '')
    return data_path 

def remember_rewards(rewards, text, epsilon):
    f2 = plt.figure()
    plt.plot(rewards)
    data = data_path()
    plt.savefig(data+f"\data_{epsilon}.png")
    plt.clf()

    f = open("rewards.txt", "r")
    s = f.read()
    s += text

    f = open("rewards.txt", "w")
    f.write(s)
    f.close()


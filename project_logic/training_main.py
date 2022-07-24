import configparser
from pyexpat import model
import Memory
if __name__ == "__main__":
    content = configparser.ConfigParser()
    content.read("project_logic\\training.ini")
    
    memory_min = content['Memory'].getint("min_size")
    memory_max = content['Memory'].getint("max_size")

    memory = Memory(max_size=memory_max, min_size=memory_min)
    model = None
    simulator = None
    #simulator.run()

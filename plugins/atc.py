from bluesky import core, traf
from bluesky.core import Entity

def init_plugin():
    config = {
        'plugin_name':     'ATC',
        'plugin_type':     'sim',
    }
    traf.mcre(5)

    return config

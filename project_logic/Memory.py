from dataclasses import dataclass
import random


@dataclass
class AirCraft:
    lat: float
    long: float
    alt: float
    heading: float
    speed: float
    aceleration: float

class Memory:
    """
        Collect and remember data for some perioud of time.
        Using this class we can train model. 
    """
    def __init__(self, max_size, min_size) -> None:
        self._samples = []
        self._max_size = max_size
        self._min_size = min_size

    def add_sample(self, sample):
        """
            Add new sample to memory.
        """
        self._samples.append(sample)
        if self._size_now() > self._max_size:
            self._samples.pop(0)

    def get_samples(self, n):
        """
            Get n samples randomly from the memory
        """
        if self._size_now() < self._min_size:
            return []
        
        if n > self._size_now():
            return random.sample(self._samples, self._size_now)
        else:
            return random.sample(self._samples, n)
        
    def _size_now(self):
        """
        Check how full the memory is
        """
        return len(self._samples)

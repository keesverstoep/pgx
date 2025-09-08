from abc import ABC, abstractmethod

class ExternalPlayer(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def play(self, state, swap_color, trace, verbose):
        pass

    @abstractmethod
    def print_valid(self, env, swap_color, trace, verbose):
        pass

    @abstractmethod
    def print_oracle_status(self, state, trace, swap_color):
        pass

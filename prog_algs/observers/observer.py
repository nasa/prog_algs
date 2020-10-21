from abc import ABC, abstractmethod, abstractproperty

class Observer(ABC):
    """
    """
    @abstractmethod
    def step(self, t, u: dict, z: dict):
        pass

    @abstractproperty
    def x(self):
        pass

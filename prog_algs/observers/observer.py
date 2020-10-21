from abc import ABC, abstractmethod

class Observer(ABC):
    """
    """
    @abstractmethod
    def step(self, t, u: dict, z: dict) -> dict:
        pass

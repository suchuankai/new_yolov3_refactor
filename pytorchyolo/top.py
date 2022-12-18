from abc import ABC, abstractmethod

class Top(ABC):

    @abstractmethod
    def _create_data_loader(self):
        pass

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def start(self):
        pass

    
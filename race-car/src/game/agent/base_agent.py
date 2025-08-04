from abc import ABC, abstractmethod

class BaseAgent(ABC):
    @abstractmethod
    def select_action(self, observation):
        pass

    @abstractmethod
    def store_transition(self, obs, action, reward, next_obs, done):
        pass

    @abstractmethod
    def learn(self):
        pass

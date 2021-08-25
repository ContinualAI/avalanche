from abc import ABC, abstractmethod


class BaseModel(ABC):
    """
    A base abstract class for models
    """

    @abstractmethod
    def get_features(self, x):
        """
        Get features from model given input
        """

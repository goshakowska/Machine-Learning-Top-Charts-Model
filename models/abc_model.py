from abc import ABC, abstractmethod


class AbstractModel(ABC):
    """
    Abstract class for all models.
    """

    @abstractmethod
    def get_parameters(self):
        """
        Get the hyperparameters suited for the model.
        """
        ...

    @abstractmethod
    def fit(self, X, y):
        """
        Fits model to the data provided.
        """
        ...


    @abstractmethod
    def predict(self, X):
        """
        Predict the class labels for the provided data
        """
        ...

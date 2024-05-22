from abc_model import AbstractModel
import numpy as np
from sklearn.linear_model import LinearRegression


class AdvancedModel(AbstractModel):

    def __init__(self, parameters):
        self.parameters = parameters
        # self.model = LinearRegression()

    def get_parameters(self):
        return self.parameters

    def fit(self, X, y):
        ...
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

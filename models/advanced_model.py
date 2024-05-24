from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import HuberRegressor
import numpy as np
from models.abc_model import AbstractModel


class AdvancedModel(AbstractModel):
    def __init__(self, parameters=None):
        self.parameters = parameters
        self.model = HuberRegressor(**(parameters if parameters else {}))
        self.scaler = StandardScaler()

    def get_parameters(self):
        return self.parameters

    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        return rmse

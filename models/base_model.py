from abc_model import AbstractModel
from pandas import DataFrame
from sklearn.linear_model import LinearRegression


class BaseModel(AbstractModel):
    def __init__(self, tracks_per_list=15, music_genre=None):

        self.parameters = {
            "tracks_per_list": tracks_per_list,
            "music_genre": music_genre
        }

        # self.parameters = parameters
        # self.model = LinearRegression()

    @staticmethod
    def validate_provided_data(X: DataFrame) -> None:
        """
        Validates the provided data by checking if it is a pandas DataFrame.
        Parameters:
            X (DataFrame): The data to be validated.
        Returns:
            None
        Raises:
            TypeError: If the provided data is not a pandas DataFrame.
        """
        required_attributes = []
        # odfiltrować reklamy
        ...

    def get_parameters(self) -> dict:
        return self.parameters
        """
        Get the parameters of the object.
        Returns:
            dict: A dictionary containing the parameters of the object.
        """

    def fit(self, X, y):
        pass

    def predict(self, X) -> dict:
        """
        Predicts the target variable for the given input data.
        Parameters:
            X (pandas.DataFrame): The input data for prediction.
        Returns:
            pandas.Series: The predicted target variable values.

        This function performs the following steps:
        1. Filters the songs based on the last 3 weeks of listening history.
        2. Sorts the songs based on the number of listenings.
        3. Truncates the bottom end of the data.

        The predicted target variable values are returned in the same format as the input data.
        """

        # filtrowanie piosenek - wybór tych słuchanych w ostatnich 3 tygodniach
        # sortowanie pod względem liczby odsłuchań
        # odcinanie dolnego końca
        ...
        return  # top_charts_dict

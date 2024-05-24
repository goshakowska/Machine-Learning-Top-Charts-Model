import pandas as pd
from models.abc_model import AbstractModel
from datetime import datetime, timedelta
from pandas import DataFrame
from src.data.make_dataset import prepare_dataset, load_data, RAW_DATA_DIR, PROCESSED_DATA_DIR
import os


class BaseModel(AbstractModel):
    def __init__(self, tracks_per_list=15, music_genre=None):

        self.parameters = {
            "tracks_per_list": tracks_per_list,
            "music_genre": music_genre
        }
        # self.parameters = parameters
        # self.model = LinearRegression()

    # @staticmethod
    # def validate_provided_data(X: DataFrame) -> None:
    #     """
    #     Validates the provided data by checking if it is a pandas DataFrame.
    #     Parameters:
    #         X (DataFrame): The data to be validated.
    #     Returns:
    #         None
    #     Raises:
    #         TypeError: If the provided data is not a pandas DataFrame.
    #     """
    #     necessary_attributes = {"track_id", "event_type", "timestamp"}
    #     if not necessary_attributes.issubset(X.columns):
    #         raise ValueError("Provided data doesn't contain necessary attributes.")

    # def process_data(self, X: DataFrame) -> DataFrame:
    #     """
    #     Processes the provided data by filtering the songs based on the last 3 weeks of listening history.
    #     Parameters:
    #         X (DataFrame): The data to be processed.
    #     Returns:
    #         None
    #     """
    #     X['timestamp'] = pd.to_datetime(X['timestamp'])
    #     three_weeks_ago = datetime.now() - timedelta(weeks=3)
    #     recent_data = X[X['timestamp'] >= three_weeks_ago]
    #     return recent_data

    def get_parameters(self) -> dict:
        """
        Get the parameters of the object.
        Returns:
            dict: A dictionary containing the parameters of the object.
        """
        return self.parameters

    def fit(self, X, y):
        pass


    def predict(self) -> dict:
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
        session_file = os.path.join(RAW_DATA_DIR, "sessions.jsonl")
        tracks_file = os.path.join(RAW_DATA_DIR, "tracks.jsonl")
        artists_file = os.path.join(RAW_DATA_DIR, "artists.jsonl")

        top_tracks_df = prepare_dataset(session_file, tracks_file, artists_file)
        final_top_list = top_tracks_df.head(self.parameters["tracks_per_list"])
        print(final_top_list)
        return final_top_list


    # def save_to_jsonl(self, df: DataFrame, file_path: str) -> None:
    #     df.to_json(file_path, orient='records', lines=True)


# def predict_with_base_model(tracks_per_list=None, music_genre=None) -> DataFrame:
#     session_file = os.path.join(RAW_DATA_DIR, "sessions.jsonl")
#     tracks_file = os.path.join(RAW_DATA_DIR, "tracks.jsonl")
#     artists_file = os.path.join(RAW_DATA_DIR, "artists.jsonl")

#     base_model = BaseModel(tracks_per_list, music_genre)
#     session_df = load_data(session_file)
#     top_tracks_df = base_model.predict(session_df, tracks_file, artists_file)
#     # base_model.save_to_jsonl(top_tracks_df, os.path.join(PROCESSED_DATA_DIR, "top_tracks.jsonl"))
#     return top_tracks_df

import pandas as pd
from models.abc_model import AbstractModel
from src.data.make_dataset import prepare_dataset, RAW_DATA_DIR
import os


class BaseModel(AbstractModel):
    """
    Initializes the BaseModel object.
    Parameters:
        tracks_per_list (int): The number of tracks per list. Defaults to 15.
        music_genre (str, optional): The genre of music. Defaults to None.
    Returns:
        None
    """
    def __init__(self, tracks_per_list=15, music_genre=None):

        self.parameters = {
            "tracks_per_list": tracks_per_list,
            "music_genre": music_genre
        }


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
        Predicts the top tracks based on the session, track, and artist data.
        Returns:
            dict: A dictionary containing the top tracks.
        """
        session_file = os.path.join(RAW_DATA_DIR, "sessions.jsonl")
        tracks_file = os.path.join(RAW_DATA_DIR, "tracks.jsonl")
        artists_file = os.path.join(RAW_DATA_DIR, "artists.jsonl")

        top_tracks_df = prepare_dataset(session_file, tracks_file, artists_file)
        final_top_list = top_tracks_df.head(self.parameters["tracks_per_list"])
        print(final_top_list)
        return final_top_list

import pandas as pd
from pandas import DataFrame
import os
from datetime import datetime, timedelta

# # Determine the base directory relative to the current file's location
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

# Define the data directories relative to the base directory
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw", "Z04_T69_V2")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")


def load_data(file_path):
    return pd.read_json(file_path, lines=True)


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
    necessary_attributes = {"track_id", "event_type", "timestamp"}
    if not necessary_attributes.issubset(X.columns):
        raise ValueError("Provided data doesn't contain necessary attributes.")


def process_data(X: DataFrame) -> DataFrame:
    """
    Processes the provided data by filtering the songs based on the last 3 weeks of listening history.
    Parameters:
        X (DataFrame): The data to be processed.
    Returns:
        None
    """
    X['timestamp'] = pd.to_datetime(X['timestamp'])
    three_weeks_ago = datetime.now() - timedelta(weeks=3)
    recent_data = X[X['timestamp'] >= three_weeks_ago]
    return recent_data


def prepare_dataset(session_file, tracks_file, artists_file):
    """
    Prepares a dataset by loading session, tracks, and artists data, validating the provided session data,
    processing the data, filtering the session data, counting the number of plays for each track,
    renaming columns in tracks and artists data, merging the dataframes, and returning a final dataframe
    with track information, play count, track name, artist id, artist name, and genres.
    Parameters:
        session_file (str): The path to the session data file.
        tracks_file (str): The path to the tracks data file.
        artists_file (str): The path to the artists data file.
    Returns:
        pandas.DataFrame: A dataframe with track information, play count, track name, artist id, artist name, and genres.
    """
    session_df = load_data(session_file)
    tracks_df = load_data(tracks_file)
    artists_df = load_data(artists_file)

    validate_provided_data(session_df)

    session_df = process_data(session_df)

    filtered_session_df = session_df[session_df['event_type'].isin(['play', 'like'])]

    track_counts = filtered_session_df['track_id'].value_counts().reset_index()
    track_counts.columns = ['track_id', 'play_count']

    tracks_df = tracks_df.rename(columns={'id': 'track_id', 'name': 'track_name'})
    artists_df = artists_df.rename(columns={'id': 'artist_id', 'name': 'artist_name'})

    track_info = track_counts.merge(tracks_df[['track_id', 'track_name', 'id_artist']], left_on='track_id', right_on='track_id', how='left')
    full_info = track_info.merge(artists_df[['artist_id', 'artist_name', 'genres']], left_on='id_artist', right_on='artist_id', how='left')

    final_df = full_info[['track_id', 'play_count', 'track_name', 'id_artist', 'artist_name', 'genres']]

    return final_df


def main():
    session_file = os.path.join(RAW_DATA_DIR, "sessions.jsonl")
    tracks_file = os.path.join(RAW_DATA_DIR, "tracks.jsonl")
    artists_file = os.path.join(RAW_DATA_DIR, "artists.jsonl")

    top_tracks_df = prepare_dataset(session_file, tracks_file, artists_file)

    top_tracks_df.to_json(os.path.join(PROCESSED_DATA_DIR, "top_tracks.jsonl"), orient='records', lines=True)


if __name__ == "__main__":
    main()

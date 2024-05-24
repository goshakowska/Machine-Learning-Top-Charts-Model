from models.advanced_model import AdvancedModel
from src.data.prepare_datasets import get_dataframe

import pandas as pd
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split

from src.models_functions.train_model import train_advanced_model


def filter_first_three_weeks(df):
    first_day_of_month = df['timestamp'].min().replace(day=1)
    last_day = first_day_of_month + timedelta(days=21)
    return df[(df['timestamp'] >= first_day_of_month) & (df['timestamp'] < last_day)]


def predict_advanced_model(tracks_per_list):
    pass
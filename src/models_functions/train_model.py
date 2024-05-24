from models.advanced_model import AdvancedModel
from src.data.prepare_datasets import get_dataframe

import pandas as pd
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split


def filter_first_three_weeks(df):
    first_day_of_month = df['timestamp'].min().replace(day=1)
    last_day = first_day_of_month + timedelta(days=21)
    return df[(df['timestamp'] >= first_day_of_month) & (df['timestamp'] < last_day)]


def train_advanced_model(tracks_per_list):
    sessions_df = get_dataframe("sessions")
    tracks_df = get_dataframe("tracks")

    merged_df = pd.merge(tracks_df, sessions_df, left_on='id', right_on='track_id')
    merged_df['timestamp'] = pd.to_datetime(merged_df['timestamp'])

    merged_df['month'] = merged_df['timestamp'].dt.month
    merged_df['year'] = merged_df['timestamp'].dt.year

    latest_year = merged_df['year'].max()
    latest_month = merged_df[merged_df['year'] == latest_year]['month'].max()

    latest_month_df = merged_df[(merged_df['year'] == latest_year) & (merged_df['month'] == latest_month)]

    first_three_weeks_df = merged_df.groupby(['year', 'month']).apply(filter_first_three_weeks).reset_index(drop=True)
    recent_plays_df = first_three_weeks_df[first_three_weeks_df['event_type'] == 'play']
    tracks_play_count = recent_plays_df.groupby('id').size().reset_index(name='num_plays')

    tracks_features_df = pd.merge(tracks_df, tracks_play_count, left_on='id', right_on='id', how='left')
    tracks_features_df['num_plays'].fillna(0, inplace=True)

    monthly_plays_df = merged_df[merged_df['event_type'] == 'play']
    tracks_monthly_play_count = monthly_plays_df.groupby('id').size().reset_index(name='monthly_plays')
    tracks_target_df = pd.merge(tracks_df, tracks_monthly_play_count, left_on='id', right_on='id', how='left')
    tracks_target_df['monthly_plays'].fillna(0, inplace=True)

    features = ['duration_ms', 'explicit', 'danceability', 'loudness', 'speechiness', 'energy', 'key', 'num_plays']
    target = 'monthly_plays'

    X = tracks_features_df[features]
    y = tracks_target_df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    best_rmse = float('inf')
    best_param = None

    for epsilon in [1.0, 1.15, 1.35, 1.5, 1.75, 2.0, 2.1, 2.2, 2.3, 10]:
        parameters = {'epsilon': epsilon}
        model = AdvancedModel(parameters)
        model.fit(X_train, y_train)
        rmse = model.evaluate(X_test, y_test)
        print(f'Epsilon: {epsilon}, RMSE: {rmse}')

        if rmse < best_rmse:
            best_rmse = rmse
            best_param = parameters

    print(f'Best parameters: {best_param}, Best RMSE: {best_rmse}')

    model = AdvancedModel(best_param)
    model.fit(X_train, y_train)

    latest_month_first_three_weeks = filter_first_three_weeks(latest_month_df)
    latest_month_play_count = latest_month_first_three_weeks[
        latest_month_first_three_weeks['event_type'] == 'play'].groupby('id').size().reset_index(name='num_plays')
    latest_month_features_df = pd.merge(tracks_df, latest_month_play_count, left_on='id', right_on='id', how='left')
    latest_month_features_df['num_plays'].fillna(0, inplace=True)

    X_latest_month = latest_month_features_df[features]
    latest_month_features_df['predicted_monthly_plays'] = model.predict(X_latest_month)

    top_tracks = latest_month_features_df.sort_values(
        by='predicted_monthly_plays', ascending=False).head(tracks_per_list)
    top_tracks["rank"] = range(1, len(top_tracks) + 1)
    print(top_tracks[['rank', 'name', 'predicted_monthly_plays']])
    return top_tracks[["rank", "name"]]

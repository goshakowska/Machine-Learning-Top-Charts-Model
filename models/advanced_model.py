from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn .linear_model import HuberRegressor
from datetime import datetime, timedelta
import pandas as pd

from models.abc_model import AbstractModel
from src.data.prepare_datasets import get_dataframe


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
        return mse


if __name__ == '__main__':
    sessions_df = get_dataframe("sessions")
    tracks_df = get_dataframe("tracks")

    merged_df = pd.merge(tracks_df, sessions_df, left_on='id', right_on='track_id')

    date_now = datetime.now()
    date_three_weeks_ago = date_now - timedelta(weeks=3)

    recent_plays_df = merged_df[(merged_df['event_type'] == 'play') & (merged_df['timestamp'] >= date_three_weeks_ago)]

    tracks_play_count = recent_plays_df.groupby('id').size().reset_index(name='num_plays')
    tracks_df = pd.merge(tracks_df, tracks_play_count, left_on='id', right_on='id', how='left')

    tracks_df['num_plays'].fillna(0, inplace=True)

    features = ['duration_ms', 'explicit', 'danceability', 'loudness', 'speechiness', 'energy', 'key', 'num_plays']
    target = 'popularity'

    X = tracks_df[features]
    y = tracks_df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    best_mse = float('inf')
    best_param = None
    for epsilon in [1.0, 1.15, 1.35, 1.5, 1.75, 2.0, 2.1, 2.2, 2.3, 10]:
        parameters = {'epsilon': epsilon}
        model = AdvancedModel(parameters)
        model.fit(X_train, y_train)
        mse = model.evaluate(X_test, y_test)
        print(f'Epsilon: {epsilon}, MSE: {mse}')
        if mse < best_mse:
            best_mse = mse
            best_param = parameters

    print(f'Best parameters: {best_param}, Best MSE: {best_mse}')

    model = AdvancedModel(best_param)
    model.fit(X_train, y_train)

    tracks_df['predicted_popularity'] = model.predict(X)
    top_20_tracks = tracks_df.sort_values(by='predicted_popularity', ascending=False).head(50)
    print(top_20_tracks[['name', 'predicted_popularity']])

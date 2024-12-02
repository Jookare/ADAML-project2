from torch.utils.data import Dataset, DataLoader
from utils import rolling_train_valid_split
import torch
import numpy as np


class TimeSeriesDataset(Dataset):
    """
    Arguments:
        df_or_path (pd.DataFrame or str): dataframe or path to the csv file
        window_size (int): size of the input sequence
        horizon (int): number of steps ahead to predict
        stride (int): stride for the windowing
    """

    def __init__(self, df, window_size=7, horizon=1, stride=1, date_as_var=True):
        self.features = df.drop(["date"], axis=1)
        self.date = df["date"]
        self.X, self.Y = self._sequencing(
            window_size=window_size,
            horizon=horizon,
            stride=stride,
            date_as_var=date_as_var,
        )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X = self.X[idx]
        Y = self.Y[idx]
        return X, Y

    def get_date(self):
        return self.date

    def _sequencing(self, window_size, horizon, stride, date_as_var):
        """
        Arguments:
            window_size (int): size of the window
            horizon (int): number of steps ahead to predict
            stride (int): stride for the windowing
        """
        n_samples, n_features = self.features.shape

        # Extract day of year
        day_of_year = self.date.dt.day_of_year.values

        sequences = range(0, n_samples - window_size - horizon + 1, stride)
        if date_as_var:
            n_features += 2

        input = np.empty((len(sequences), window_size, n_features), dtype=np.float32)
        target = np.empty((len(sequences),), dtype=np.float32)

        for i, seq_start in enumerate(sequences):
            # Combine features with day of year
            window_features = self.features.iloc[
                seq_start : seq_start + window_size, :
            ].values
            window_day_of_year = day_of_year[
                seq_start : seq_start + window_size
            ].reshape(-1, 1)

            # Combine features with day of year
            input[i, :, :4] = window_features
            if date_as_var:
                input[i, :, 4] = np.sin(
                    window_day_of_year * (2 * np.pi) / 366
                ).squeeze()
                input[i, :, 5] = np.cos(
                    window_day_of_year * (2 * np.pi) / 366
                ).squeeze()

            target[i] = self.features["humidity"].iloc[
                seq_start + window_size + horizon - 1
            ]
        return torch.from_numpy(input), torch.from_numpy(target)

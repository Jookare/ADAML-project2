import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset


class TimeSeriesDataset(Dataset):
    """
    Arguments:
        df_or_path (pd.DataFrame or str): dataframe or path to the csv file
        window_size (int): size of the input sequence
        horizon (int): number of steps ahead to predict
        stride (int): stride for the windowing
    """

    def __init__(self, df_or_path=None, window_size=7, horizon=1, stride=1):

        if isinstance(df_or_path, str):
            df = pd.read_csv(df_or_path)
        elif isinstance(df_or_path, pd.DataFrame):
            df = df_or_path
        else:
            raise ValueError(
                f"'df_or_path' must be a pandas DataFrame or a string, got {type(df_or_path)}"
            )
        self.features = df.drop(["date"], axis=1)
        self.date = df["date"]
        self.X, self.Y = self._sequencing(
            window_size=window_size, horizon=horizon, stride=stride
        )

    def __len__(self):
        return len(self.date)

    def __getitem__(self, idx):
        X = self.X[idx]
        Y = self.Y[idx]
        return X, Y

    def get_date(self):
        return self.date

    def _sequencing(self, window_size, horizon, stride=1):
        """
        Arguments:
            window_size (int): size of the window
            horizon (int): number of steps ahead to predict
            stride (int): stride for the windowing
        """
        n_samples, n_features = self.features.shape
        sequences = range(0, n_samples - window_size - horizon + 1, stride)
        inp, target = np.empty((len(sequences), window_size, n_features)), np.empty(
            (len(sequences), horizon)
        )
        for i in sequences:
            inp[i, :, :] = self.features.iloc[i : i + window_size, :].values
            target[i, :] = (
                self.features["humidity"]
                .iloc[i + window_size : i + window_size + horizon]
                .values
            )
        return torch.from_numpy(inp), torch.from_numpy(target)


def train_model(model, dataloader, criterion, optimizer, num_epochs=25, device="cpu"):
    """
    Arguments:
        model (torch.nn.Module): model to train
        dataloader (torch.utils.data.DataLoader): dataloader for the dataset
        criterion (torch.nn.Module): loss function
        optimizer (torch.optim.Optimizer): optimizer
        num_epochs (int): number of epochs to train
        device (torch.device): device to run the model on
    """
    losses = []
    val_losses = []
    for epoch in range(num_epochs):
        model.train()
        loss_over_epoch = 0.0
        for inputs, targets in dataloader:
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Print statistics
            loss_over_epoch += loss.item()
        print(
            f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss_over_epoch/len(dataloader)}"
        )
        loss.append(loss_over_epoch / len(dataloader))
        validation_loss = evaluation(model, criterion, dataloader, device)
        val_losses.append(validation_loss)
    return losses, val_losses


def evaluation(model, criterion, dataloader, device):
    """
    Arguments:
        model (torch.nn.Module): model to evaluate
        criterion (torch.nn.Module): loss function
        dataloader (torch.utils.data.DataLoader): dataloader for the dataset
        device (torch.device): device to run the model on
    """
    model.eval()  # Switch to evaluation mode
    with torch.no_grad():  # No .backward() used so switch to no_grad context
        loss = 0.0  # validation/test loss
        for input, target in dataloader:
            input, target = input.to(device), target.to(device)
            output = model(input)
            loss += criterion(output, target).item()

    return loss / len(dataloader)  # average loss over the dataset

import torch
import pandas as pd
from torch.utils.data import Dataset


class TimeSeriesDataset(Dataset):
    """
    Arguments:
        path (string): path of the csv file (assuming the csv file has a column named "date" and a column named "target")

    """

    def __init__(self, path):
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"])
        self.X = df.drop(["date"], ["target"], axis=1)
        self.date = df["date"]
        self.Y = df["target"]

    def __len__(self):
        return len(self.date)

    def __getitem__(self, idx):
        X = torch.tensor(self.X.iloc[idx, :].values)
        Y = torch.tensor(self.Y.iloc[idx])
        return X, Y

    def get_date(self):
        return self.date


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

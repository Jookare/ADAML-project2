import torch
import torch.nn as nn
import numpy as np

# Train model
def train(model, train_loader, criterion, optimizer, device, model_type):
    # Set model to train
    model.train()
    total_train_loss = 0
    # Iterate over the batches
    for batch in train_loader:
        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device)

        if model_type == "RNN" or model_type == "LSTM":
            # Initialize the hidden state
            prev_state = model.init_state(inputs.size(0), device)
        
            # Pass the input through the model
            outputs, _ = model(inputs, prev_state)
        else:
            input = inputs[:, :-1, :]
            target = inputs[:, 1:, :]
            outputs = model(input, target)
            
        # Squeeze away the extra dimensions
        outputs = outputs.squeeze()
        targets = targets.squeeze()
        
        # Calculate loss
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        
        optimizer.step()
        
        # Accumulate the training loss
        total_train_loss += loss.item()

    # Return average train loss
    avg_train_loss = total_train_loss / len(train_loader)
    return avg_train_loss


def validate(model, valid_loader, criterion, device, model_type):
    # Set model to validation
    model.eval()
    total_val_loss = 0
    
    with torch.no_grad(): # Iterate
        for batch in valid_loader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)
            if model_type == "RNN" or model_type == "LSTM":
                # Initialize the hidden state
                prev_state = model.init_state(inputs.size(0), device)
            
                # Pass the input through the model
                outputs, _ = model(inputs, prev_state)
            else:
                input = inputs[:, :-1, :]
                target = inputs[:, 1:, :]
                outputs= model(input, target)
        
            # Squeeze away the extra dimensions
            outputs = outputs.squeeze()
            targets = targets.squeeze()
            loss = criterion(outputs, targets)
            
            # Calculate loss
            total_val_loss += loss.item()
        
    # Return average validation loss
    avg_val_loss = total_val_loss / len(valid_loader)
    return avg_val_loss

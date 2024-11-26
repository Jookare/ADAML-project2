import torch
import torch.nn as nn

class RNN_model(nn.Module):
    def __init__(self, hidden_size, embed_dim=128, input_size=6, n_layers=1, out_features=1, dropout_p=0.2):
        super(RNN_model, self).__init__()

        self.input_size = input_size
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p

        
        self.embed = nn.Linear(in_features=self.input_size, out_features=self.embed_dim)

        # dimensions: batches x seq_length x emb_dim
        self.rnn = nn.RNN(input_size=self.embed_dim, hidden_size=self.hidden_size,
                          num_layers=self.n_layers, dropout=self.dropout_p, batch_first=True)

        self.dropout = nn.Dropout(self.dropout_p)


        self.fc = nn.Linear(in_features=self.hidden_size,
                            out_features=out_features)
        
    def forward(self, input, prev_state):

        input = self.embed(input)
        output, state = self.rnn(input, prev_state)

        # Output shape is [batch x sentence_length x hidden_size]
        output = self.dropout(output[:, -1, :])
        output = self.fc(output)  # Shape: [batch_size, num_classes]
        return output, state

    def init_state(self, batch_size, device):
        h0 = torch.zeros((self.n_layers, batch_size, self.hidden_size)).to(device)
        return h0

class LSTM_model(nn.Module):
    def __init__(self, hidden_size, embed_dim=128, input_size=6, n_layers=1, out_features=1, dropout_p=0.2):
        super(LSTM_model, self).__init__()
        
        self.input_size = input_size
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p

        self.embed = nn.Linear(in_features=self.input_size, out_features=self.embed_dim)

        # define lstm
        self.lstm = nn.LSTM(
            input_size = self.embed_dim,
            hidden_size = self.hidden_size,
            num_layers = self.n_layers,
            dropout=dropout_p,
            batch_first = True
        )
        
        # define dropout
        self.dropout = nn.Dropout(p=self.dropout_p)

        # define linear
        self.fc = nn.Linear(in_features=self.hidden_size,
                            out_features=out_features)

    def forward(self, input, prev_state):

        input = self.embed(input)
        # yhat is the full sequence prediction, while state is the last hidden state (coincides with yhat[-1] if n_layers=1)
        output, state = self.lstm(input, prev_state)   

        output = self.dropout(output[:, -1, :])
        output = self.fc(output)
        return output, state

    def init_state(self, batch_size, device):
        # return tuple
        
        h0 = torch.zeros((self.n_layers, batch_size, self.hidden_size)).to(device)
        c0 = torch.zeros((self.n_layers, batch_size, self.hidden_size)).to(device)
        return (h0,c0)
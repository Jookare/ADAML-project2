import torch
import torch.nn as nn

class RNN_model(nn.Module):
    def __init__(self, hidden_size, input_size=4, n_layers=1, out_features=1, dropout_p=0.2):
        super(RNN_model, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p

        # dimensions: batches x seq_length x emb_dim
        self.rnn = nn.RNN(input_size=self.input_size, hidden_size=self.hidden_size,
                          num_layers=self.n_layers, dropout=dropout_p, batch_first=True)

        self.dropout = nn.Dropout(self.dropout_p)

        self.fc = nn.Linear(in_features=self.hidden_size,
                            out_features=out_features)
        
    def forward(self, input, prev_state):
        output, state = self.rnn(input, prev_state)

        # Output shape is [batch x sentence_length x hidden_size]
        output = self.dropout(output[:, -1, :])
        output = self.fc(output)  # Shape: [batch_size, num_classes]
        return output, state

    def init_state(self, batch_size, device):
        h0 = torch.zeros((self.n_layers, batch_size, self.hidden_size)).to(device)
        return h0

import torch.nn as nn


# Receiver: generate log probabilities over possible sums from embedded input symbol
class Receiver(nn.Module):

    def __init__(self, input_dim, n_sums, n_layers, hidden_dim=None):
        super(Receiver, self).__init__()

        self.layers = nn.ModuleList()
        for k in range(n_layers):
            if k == 0:
                dim1 = input_dim
            else:
                dim1 = hidden_dim
            if k == n_layers-1:
                dim2 = n_sums
            else:
                dim2 = hidden_dim
            self.layers.append(nn.Linear(dim1, dim2))
            if k < n_layers-1:
                self.layers.append(nn.ReLU())

    def forward(self, x, _input, _aux_input):
        for layer in self.layers:
            x = layer(x)
        return x


# Sender: generate log probabilities over vocabulary
class Sender(nn.Module):

    def __init__(self, n_features, n_symbols, n_layers, hidden_dim=None):
        super(Sender, self).__init__()

        self.layers = nn.ModuleList()
        for k in range(n_layers):
            if k == 0:
                dim1 = n_features
            else:
                dim1 = hidden_dim
            if k == n_layers-1:
                dim2 = n_symbols
            else:
                dim2 = hidden_dim
            self.layers.append(nn.Linear(dim1, dim2))
            if k < n_layers-1:
                self.layers.append(nn.ReLU())

    def forward(self, x, _aux_input):
        for layer in self.layers:
            x = layer(x)
        return x

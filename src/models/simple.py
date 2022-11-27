import torch
from torch import nn


class simple_NN(nn.Module):
    def __init__(self, num_features, num_output):
        super(simple_NN, self).__init__()
        self.num_features = num_features
        self.num_classes = num_output
        self.dropout = nn.Dropout1d(0.2)
        self.activation = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

        # Define the network
        self.network = nn.Sequential(
            nn.Linear(self.num_features, 64),
            self.dropout,
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_output),
        )

        # self.apply(self._init_weights)

    def forward(self, x):
        logits = self.network(x)
        return self.softmax(logits)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()


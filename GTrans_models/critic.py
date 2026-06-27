import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPCritic(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=4, dropout=0.1):
        """
        num_layers: total number of layers (input -> hidden*(L-2) -> output), must >= 2
        """
        super(MLPCritic, self).__init__()
        assert num_layers >= 2, "At least 2 layers required"

        layers = []


        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))


        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))


        layers.append(nn.Linear(hidden_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)





















































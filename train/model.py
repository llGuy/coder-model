import torch
import torch.nn as nn
import torch.nn.functional as F

# This defines a simple multi-layer network
class MultiLayerNet(nn.Module):
    def __init__(
        self, 
        input_dim: int,
        layer_dims: list[int],
        output_dim: int
    ):
        super(MultiLayerNet, self).__init__()

        self.fc = []

        prev_dim = input_dim
        for hidden_dim in layer_dims:
            self.fc.append(nn.Linear(prev_dim, hidden_dim))
            prev_dim = hidden_dim

        self.fc.append(nn.Linear(prev_dim, output_dim))

    def forward(self, x):
        for i in range(len(self.fc)):
            x = self.fc[i](x)

            # Only apply non-linearity on layers before the last
            if i != (len(self.fc) - 1):
                x = F.relu(x)

        return x

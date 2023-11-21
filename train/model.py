import torch
import torch.nn as nn
import torch.nn.functional as F

import data

class BaselineNet(nn.Module):
  def __init__(self):
    super(Net, self).__init__()

    # Fully connected hidden layers
    self.fc1 = nn.Linear(data.INPUT_DIM, 512)
    self.fc2 = nn.Linear(512, 128)

    # Then, we go through the branches
    self.br1 = nn.Linear(128, data.BRANCH1_OUTPUTS)
    self.br2 = nn.Linear(128, data.BRANCH2_OUTPUTS)
    self.br3 = nn.Linear(128, data.BRANCH3_OUTPUTS)

  # Returns a tuple with the outputs from the 3 branches
  def forward(self, x):
    # First hidden layer
    x = self.fc1(x)
    x = F.relu(x)

    # Second hidden layer
    x = self.fc2(x)
    x = F.relu(x)

    # Branch out

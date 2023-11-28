"""
Useful link for pytorch training:

https://pytorch.org/tutorials/beginner/introyt/trainingyt.html
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import data

class BaselineNet(nn.Module):
    def __init__(self):
        super(BaselineNet, self).__init__()

        # Fully connected hidden layers
        self.fc1 = nn.Linear(data.INPUT_DIM, 512)
        self.fc2 = nn.Linear(512, 64)

        # Then, we go through the branches
        self.br1 = nn.Linear(64, data.BRANCH1_OUTPUTS)
        self.br2 = nn.Linear(64, data.BRANCH2_OUTPUTS)
        self.br3 = nn.Linear(64, data.BRANCH3_OUTPUTS)

    # Returns a tuple with the outputs from the 3 branches
    def forward(self, x):
        # First hidden layer
        x = self.fc1(x)
        x = F.relu(x)

        # Second hidden layer
        x = self.fc2(x)
        x = F.relu(x)

        # Branch out
        b1 = self.br1(x)
        b1 = b1.reshape((*b1.shape[:-1], data.INSTRUCTIONS_PER_PROGRAM, 5))
        out1 = F.log_softmax(b1, dim=len(b1.shape)-1)

        b2 = self.br2(x)
        b2 = b2.reshape((*b2.shape[:-1], data.INSTRUCTIONS_PER_PROGRAM, 6))
        out2 = F.log_softmax(b2, dim=len(b2.shape)-1)

        b3 = self.br3(x)
        b3 = b3.reshape((*b3.shape[:-1], data.INSTRUCTIONS_PER_PROGRAM, 26))
        out3 = F.log_softmax(b3, dim=len(b3.shape)-1)

        return out1, out2, out3   

# This one doesn't work...
class ConvolutionNet(nn.Module):
    def __init__(self):
        super(ConvolutionNet, self).__init__()

        # Convolution nets first
        self.cnn1 = nn.Conv1d(1, 1, 12)
        self.fc = nn.Linear(1 * (1))

        # Then, we go through the branches
        self.br1 = nn.Linear(64, data.BRANCH1_OUTPUTS)
        self.br2 = nn.Linear(64, data.BRANCH2_OUTPUTS)
        self.br3 = nn.Linear(64, data.BRANCH3_OUTPUTS)

    def forward(self, x):
        print(x.shape)

        # First conv hidden layer
        x = self.cnn1(x)
        x = F.relu(x)

        # Second conv hidden layer
        x = self.cnn2(x)
        x = F.relu(x)

        # Linear hidden layer
        x = self.cnn2(x)
        x = F.relu(x)

        # Branch out
        b1 = self.br1(x)
        b1 = b1.reshape((*b1.shape[:-1], data.INSTRUCTIONS_PER_PROGRAM, 5))
        out1 = F.log_softmax(b1, dim=len(b1.shape)-1)

        b2 = self.br2(x)
        b2 = b2.reshape((*b2.shape[:-1], data.INSTRUCTIONS_PER_PROGRAM, 6))
        out2 = F.log_softmax(b2, dim=len(b2.shape)-1)

        b3 = self.br3(x)
        b3 = b3.reshape((*b3.shape[:-1], data.INSTRUCTIONS_PER_PROGRAM, 26))
        out3 = F.log_softmax(b3, dim=len(b3.shape)-1)

# This one is a many to one network
class RecurrentNeuralNet1(nn.Module):
    def __init__(self):
        super(RecurrentNeuralNet1, self).__init__()

        self.hidden_dim = 64

        self.rnn = nn.RNN(data.INTS_PER_IO_PAIR, self.hidden_dim, batch_first=True)
        
        self.br1 = nn.Linear(64, data.BRANCH1_OUTPUTS)
        self.br2 = nn.Linear(64, data.BRANCH2_OUTPUTS)
        self.br3 = nn.Linear(64, data.BRANCH3_OUTPUTS)

    def forward(self, x):
        batch_size = x.shape[0]

        h0 = torch.zeros((1, batch_size, self.hidden_dim))

        rnn_out, _ = self.rnn(x, h0)

        rnn_out = rnn_out[:, -1, :]

        # Branch out
        b1 = self.br1(rnn_out)
        b1 = b1.reshape((*b1.shape[:-1], data.INSTRUCTIONS_PER_PROGRAM, 5))
        out1 = F.log_softmax(b1, dim=len(b1.shape)-1)

        b2 = self.br2(rnn_out)
        b2 = b2.reshape((*b2.shape[:-1], data.INSTRUCTIONS_PER_PROGRAM, 6))
        out2 = F.log_softmax(b2, dim=len(b2.shape)-1)

        b3 = self.br3(rnn_out)
        b3 = b3.reshape((*b3.shape[:-1], data.INSTRUCTIONS_PER_PROGRAM, 26))
        out3 = F.log_softmax(b3, dim=len(b3.shape)-1)

        return out1, out2, out3

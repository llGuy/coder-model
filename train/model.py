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
   
# Given a batch, perform SGD.
def train(model, dataloader):

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    io_pairs_batch, prog_labels_batch = next(iter(dataloader))
    N, _ = io_pairs_batch.shape

    print(io_pairs_batch.size())
    print(prog_labels_batch[0].size())
    print(prog_labels_batch[1].size())
    print(prog_labels_batch[2].size())


    return

    for i in range(N):
        optimizer.zero_grad()
        out1, out2, out3 = model(io_pairs_batch[i])

        # Get loss and gradients for each branch.
        loss1 = torch.nn.CrossEntropyLoss(out1, prog_labels_batch[0, i])
        loss1.backward()

        loss2 = torch.nn.CrossEntropyLoss(out2, prog_labels_batch[1, i])
        loss2.backward()

        loss3 = torch.nn.CrossEntropyLoss(out3, prog_labels_batch[2, i])
        loss3.backward()

        # Adjust learning weights.
        optimizer.step()

        print(loss1, loss2, loss3)


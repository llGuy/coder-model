import torch
from torch.utils.data import DataLoader

import model
import data

# Given a batch, perform SGD.
def train(model, dataloader):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    loss_func = torch.nn.L1Loss()

    for b_idx, (b_inputs, b_labels) in enumerate(dataloader):
        optimizer.zero_grad()
        out1, out2, out3 = model(b_inputs)

        # Get loss and gradients for each branch.
        loss1 = loss_func(out1, b_labels[0])

        loss2 = loss_func(out2, b_labels[1])

        loss3 = loss_func(out3, b_labels[2])

        total_loss = loss1 + loss2 + loss3
        total_loss.backward()

        print(total_loss.item())

        # Adjust learning weights.
        optimizer.step()


def main():
    trainset = data.ProgDataset('../data/dataset')
    dl = DataLoader(trainset, batch_size=50, shuffle=True)

    bn = model.BaselineNet()
    train(bn, dl)

    pass

if __name__ == "__main__":
    main()

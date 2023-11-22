import torch
from torch.utils.data import DataLoader

import model
import data

import argparse

NUM_TRAIN_EXAMPLES = 10000
NUM_VAL_EXAMPLES = 100

def get_val_loss(model, dl_val, loss_func):
    running_loss = 0

    model.eval()
    with torch.no_grad():
        for b_idx, (b_inputs, b_labels) in enumerate(dl_val):
            out1, out2, out3 = model(b_inputs)

            loss1 = loss_func(out1, b_labels[0])
            loss2 = loss_func(out2, b_labels[1])
            loss3 = loss_func(out3, b_labels[2])

            total_loss = loss1 + loss2 + loss3

            running_loss += total_loss.item()
    return running_loss

# Given a batch, perform SGD.
def train(model, dl_train, dl_val):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.00001, momentum=0.5)

    loss_func = torch.nn.CrossEntropyLoss()

    for _ in range(5):
        for b_idx, (b_inputs, b_labels) in enumerate(dl_train):
            model.train()

            optimizer.zero_grad()
            out1, out2, out3 = model(b_inputs)

            # Get loss and gradients for each branch.
            loss1 = loss_func(out1, b_labels[0])
            loss2 = loss_func(out2, b_labels[1])
            loss3 = loss_func(out3, b_labels[2])

            total_loss = loss1 + loss2 + loss3
            total_loss.backward()

            # Adjust learning weights.
            optimizer.step()

            val_loss = get_val_loss(model, dl_val, loss_func)

            print("Train loss: ", total_loss.item(), " \t|\t Validation loss: ", val_loss)

def model_path(lr, rho):
    return 'models/model-lr' + str(lr) + '-rho' + str(rho)

def train_main(lr, rho):
    trainset = data.ProgDataset('../data/dataset/train', NUM_TRAIN_EXAMPLES)
    dl_train = DataLoader(trainset, batch_size=50, shuffle=True)

    valset = data.ProgDataset('../data/dataset/validation', NUM_VAL_EXAMPLES)
    dl_val = DataLoader(valset, batch_size=NUM_VAL_EXAMPLES, shuffle=False)

    bn = model.BaselineNet()

    train(bn, dl_train, dl_val)

    torch.save(bn, model_path(lr, rho))


def test_main(model_path, example_idx):
    dataset = data.ProgDataset('../data/dataset/validation', NUM_VAL_EXAMPLES)

    io_pairs = dataset.load_io_pairs(example_idx)

    # Test output of the model
    model = torch.load(model_path)

    # Turn off gradient computation
    model.eval()
    with torch.no_grad():
        out1, out2, out3 = model(io_pairs)

        print(out1.shape)
        print(out2.shape)
        print(out3.shape)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=str, required=True)

    # If run==train
    parser.add_argument('--lr', type=float, required=False)
    parser.add_argument('--rho', type=float, required=False)

    # If run==test
    parser.add_argument('--model', type=str, required=False)
    parser.add_argument('--example', type=int, required=False)
    args = parser.parse_args()

    if args.run == 'train':
        train_main(args.lr, args.rho)
    elif args.run == 'test':
        test_main(args.model, args.example)
    else:
        print('invalid run type (only support train and test)')

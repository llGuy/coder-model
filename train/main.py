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



# Perform SGD 5 times on 5 random batches.
def train(model, dl_train, dl_val, lr, momentum, epoch):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    loss_func = torch.nn.CrossEntropyLoss()
    # loss_func = torch.nn.MSELoss()

    for _ in range(epoch):
        batched_io_pairs, batched_labels = next(iter(dl_train))
        N, _ = batched_io_pairs.size()
        for i in range(N):
            model.train()

            optimizer.zero_grad()
            out1, out2, out3 = model(batched_io_pairs[i])

            # Get loss and gradients for each branch.
            loss1 = loss_func(out1, batched_labels[0][i])
            loss2 = loss_func(out2, batched_labels[1][i])
            loss3 = loss_func(out3, batched_labels[2][i])

            total_loss = loss1 + loss2 + loss3
            total_loss.backward()

            # Adjust learning weights.
            optimizer.step()

            val_loss = get_val_loss(model, dl_val, loss_func)

            print("Train loss: ", total_loss.item(), " \t|\t Validation loss: ", val_loss)



def model_path(lr, rho):
    return 'models/model-lr' + str(lr) + '-rho' + str(rho)


def train_main(lr, rho, batch_size, epoch):

    if lr is None:
        lr = 0.000001
    if rho is None:
        rho = 0.9
    if batch_size is None:
        batch_size = 50
    if epoch is None:
        epoch = 10

    trainset = data.ProgDataset('../data/dataset/train', NUM_TRAIN_EXAMPLES)
    dl_train = DataLoader(trainset, batch_size=batch_size, shuffle=True)

    valset = data.ProgDataset('../data/dataset/validation', NUM_VAL_EXAMPLES)
    dl_val = DataLoader(valset, batch_size=NUM_VAL_EXAMPLES, shuffle=False)

    bn = model.BaselineNet()

    train(bn, dl_train, dl_val, lr, rho, epoch)
    torch.save(bn, model_path(lr, rho))


def translate_probabilities(out1, out2, out3):
    operations = [ 'add', 'sub', 'mul', 'div', 'mov' ]
    inputs = [ 'x0', 'x1', 'x2' ]
    registers = [ 'r0', 'r1', 'r2' ]

    output = ''

    for i in range(data.INSTRUCTIONS_PER_PROGRAM):
        instruction_probs = out1[i]
        lvalue_probs = out2[i]
        rvalue_probs = out3[i]

        op_idx = int(torch.argmax(instruction_probs))
        lvalue_idx = int(torch.argmax(lvalue_probs))
        rvalue_idx = int(torch.argmax(rvalue_probs))

        instruction_str = operations[op_idx]
        lvalue_str = None
        rvalue_str = None

        if lvalue_idx < 3:
            lvalue_str = inputs[lvalue_idx]
        else:
            lvalue_str = registers[lvalue_idx - 3]

        if rvalue_idx < 3:
            rvalue_str = inputs[rvalue_idx];
        elif rvalue_idx < 6:
            rvalue_str = registers[rvalue_idx - 3]
        else:
            rvalue_str = str(rvalue_idx - 6)

        output += instruction_str + ' ' + lvalue_str + ' ' + rvalue_str + '\n'

    return output



def test_main(model_path, example_idx):
    dataset = data.ProgDataset('../data/dataset/validation', NUM_VAL_EXAMPLES)

    io_pairs = dataset.load_io_pairs(example_idx)
    label1, label2, label3 = dataset.load_label(example_idx)

    # Test output of the model
    model = torch.load(model_path)

    # Turn off gradient computation
    model.eval()
    with torch.no_grad():
        out1, out2, out3 = model(io_pairs)
        print("Predicted:")
        print(translate_probabilities(out1, out2, out3))

        print("Original:")
        print(translate_probabilities(label1, label2, label3))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=str, required=True)

    # If run==train
    parser.add_argument('--lr', type=float, required=False)
    parser.add_argument('--rho', type=float, required=False)
    parser.add_argument('--epoch', type=int, required=False)
    parser.add_argument('--batch_size', type=int, required=False)

    # If run==test
    parser.add_argument('--model', type=str, required=False)
    parser.add_argument('--example', type=int, required=False)
    args = parser.parse_args()

    if args.run == 'train':
        train_main(args.lr, args.rho, args.batch_size, args.epoch)
    elif args.run == 'test':
        test_main(args.model, args.example)
    else:
        print('invalid run type (only support train and test)')

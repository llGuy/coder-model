import torch
from torch.utils.data import DataLoader

import model
import data

import argparse

import json
import datetime


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
def train(model, dl_train, dl_val, lr, momentum, num_epochs):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    loss_func = torch.nn.CrossEntropyLoss()

    final_loss = None

    for epoch in range(num_epochs):
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

            final_loss = val_loss

            print(f"Epoch: {epoch} | Batch: {b_idx} | Train loss: {total_loss.item()} | Validation loss: {val_loss}")

    return final_loss



def model_path(final_loss, model_type):
    current_datetime = datetime.datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")

    file_name = f"models/model_{model_type}_{formatted_datetime}_loss_{final_loss}"
    return file_name



def train_main(model_type, lr, rho, batch_size, epoch):
    if model_type is None:
        model_type = 'baseline'
    if lr is None:
        lr = 0.000001
    if rho is None:
        rho = 0.9
    if batch_size is None:
        batch_size = 50
    if epoch is None:
        epoch = 10

    group_ints = (model_type == 'rnn1')

    trainset = data.ProgDataset('../data/dataset/train', data.NUM_TRAIN_EXAMPLES, group_ints)
    dl_train = DataLoader(trainset, batch_size=batch_size, shuffle=True)

    valset = data.ProgDataset('../data/dataset/validation', data.NUM_VAL_EXAMPLES, group_ints)
    dl_val = DataLoader(valset, batch_size=data.NUM_VAL_EXAMPLES, shuffle=False)

    net = None

    if model_type == 'baseline':
        net = model.BaselineNet()
    elif model_type == 'conv':
        net = model.ConvolutionNet()
    elif model_type == 'rnn1':
        net = model.RecurrentNeuralNet1()
    else:
        assert(False)

    final_loss = train(net, dl_train, dl_val, lr, rho, epoch)

    file_name = model_path(final_loss, model_type)
    torch.save(net, file_name)

    json_dict = {
        "learning_rate" : lr,
        "momentum" : rho,
        "batch_size" : batch_size,
        "num_epochs" : epoch,
        "model_type" : model_type
    }

    json_object = json.dumps(json_dict, indent=4)

    with open(file_name + ".json", "w") as outfile:
        outfile.write(json_object)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # If run==train
    parser.add_argument('--type', type=str, required=True)
    parser.add_argument('--lr', type=float, required=True)
    parser.add_argument('--rho', type=float, required=True)
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    args = parser.parse_args()

    train_main(args.type, args.lr, args.rho, args.batch_size, args.epochs)

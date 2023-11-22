from torch.utils.data import DataLoader

import model
import data

def main():
    trainset = data.ProgDataset('../data/dataset')
    dl = DataLoader(trainset, batch_size=50, shuffle=True)

    bn = model.BaselineNet()
    model.train(bn, dl)

    pass

if __name__ == "__main__":
    main()

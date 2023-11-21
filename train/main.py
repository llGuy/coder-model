import model
import data

from tqdm import tqdm

NUM_EXAMPLES = 1000

def train():
    train_data = data.Dataset("../data/dataset", NUM_EXAMPLES)
    train_loader = data.BatchedDataLoader(train_data, 100, False)

    net = model.BaselineNet()

    for batch_idx, (inputs, labels) in enumerate(tqdm(train_loader)):
        # Do forward pass, backprop and update weights.
        pass

if __name__ == "__main__":
  print("Hello world")
  train()

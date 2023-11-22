"""
This file encapsulates the process of pulling data from storage and exposing it
to the training loop. The Dataset is responsible for processing single instances of 
data. The DataLoader is responsible for collecting multiple data instances, and
returning them in a batch.

Creating a custom dataset object for your data, and using the torch dataloader:
https://pytorch.org/tutorials/beginner/basics/data_tutorial.html

More informaiton about training with pytorch:
https://pytorch.org/tutorials/beginner/introyt/trainingyt.htmlo
"""

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import numpy as np
import struct

import main

NUM_EXAMPLES = 2000

# Dimensions of the input to the model (1000 IO pairs)
INPUT_DIM = 1000 * 6 + 2

INSTRUCTIONS_PER_PROGRAM = 30

IO_PAIRS_PER_PROGRAM = 1000
INTS_PER_IO_PAIR = 6

# Baseline model dimensions
BRANCH1_OUTPUTS = INSTRUCTIONS_PER_PROGRAM * 5
BRANCH2_OUTPUTS = INSTRUCTIONS_PER_PROGRAM * 6
BRANCH3_OUTPUTS = INSTRUCTIONS_PER_PROGRAM * 26


# Custom Dataset object extending pytorch Dataset.
# This class is not responsible for datachaching or batch ops.
class ProgDataset(Dataset):

    # Init loads the entire dataset into memory
    def __init__(
        self, 
        dataset_path: str,
    ):
        self.dataset_path = dataset_path
        self.num_examples = NUM_EXAMPLES

    def __len__(self):
        return self.num_examples

    # Load label program.
    def load_label(self, idx: int):
        label_path = self.dataset_path + "/src-" + str(idx)

        dims = [5, 6, 26]

        # Parse data into lists, which can be formatted into tensors.
        data = [[] for _ in dims]

        with open(label_path, 'rb') as f:

            for i in range(INSTRUCTIONS_PER_PROGRAM):

                instr_probs = np.fromfile(f, dtype=np.float32, count=dims[0])
                lval_probs = np.fromfile(f, dtype=np.float32, count=dims[1])
                rval_probs = np.fromfile(f, dtype=np.float32, count=dims[2])
                
                if len(instr_probs) == 0 or len(lval_probs) == 0 or len(rval_probs) == 0:
                    print("Improperly formatted input file")
                    return 0

                data[0].append(instr_probs)
                data[1].append(lval_probs)
                data[2].append(rval_probs)

        data = tuple(torch.from_numpy(np.array(data[i])) for i in range(len(dims)))
        return data

    # load io-pairs for a given program.
    def load_io_pairs(self, idx: int):

        io_pair_path = self.dataset_path + "/io-pair-" + str(idx)
        output_tensor = np.empty(2 + IO_PAIRS_PER_PROGRAM * INTS_PER_IO_PAIR, dtype=np.int32)

        with open(io_pair_path, 'rb') as f:
            output_tensor[0] = int.from_bytes(f.read(1), "little", signed=False)
            output_tensor[1] = int.from_bytes(f.read(1), "little", signed=False)

            for i in range(IO_PAIRS_PER_PROGRAM * INTS_PER_IO_PAIR):
                output_tensor[2 + i] = int.from_bytes(f.read(4), "little", signed=True)

        return torch.from_numpy(output_tensor)

    def __getitem__(self, idx):
        return self.load_io_pairs(idx), self.load_label(idx)
        

# Example usage.
if __name__ == "__main__":

    # Create dataset to traverse.
    dataset_before_train_val_split = ProgDataset('../data/dataset')

    # Instantiate dataloader.
    dl = DataLoader(dataset_before_train_val_split, batch_size=100, shuffle=True)

    # Use the dataloader to iterate through the data.
    io_pairs, prog_labels = next(iter(dl))

    print(f'io_pairs batch shape: {io_pairs.size()}')
    print(f'prob_labels batch shape: {prog_labels[0].size()}, {prog_labels[1].size()}, {prog_labels[2].size()}')


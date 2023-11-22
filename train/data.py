"""
This file encapsulates the process of pulling data from storage and exposing it
to the training loop. The Dataset is responsible for processing single instances of 
data. The DataLoader is responsible for collecting multiple data instances, and
returning them in a batch.

https://pytorch.org/tutorials/beginner/introyt/trainingyt.htmlo

Creating a custom dataset and dataloader object for your data:

https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
"""

import torch
from torch.utils.data import Dataset

import numpy as np
import struct

import main

# Dimensions of the input to the model (1000 IO pairs)
INPUT_DIM = 1000 * 6

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
        self.num_examples = main.NUM_EXAMPLES

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

        data = [torch.from_numpy(np.array(data[i])) for i in range(len(dims))]
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
        
    
if __name__ == "__main__":
    obj = ProgDataset('../data/dataset')
    obj.__getitem__(0)


class BatchedDataLoader:

    # Define the splits by a dictionary with entries for 'train',
    # 'validation' and 'test'. Need to associate these with their
    # sizes.
    def __init__(
        self,
        data_set: Dataset,
        batch_size: int,
        shuffle: bool
    ):
        self.data_set = data_set
        self.batch_size = batch_size

        self.current = 0
        self.indices = None

        if shuffle:
          # TODO: implement shuffle
          pass
        else:
          self.indices = torch.arange(0, self.data_set.num_examples, 1)

    def __len__(self):
        return self.data_set.num_examples

    def __getitem__(self, idx):
        io_pairs = []
        labels = []
        for i in range(self.batch_size):
            example_idx = self.indices[i + idx * self.batch_size]
            current_io_pair = self.data_set.load_io_pair(example_idx)
            current_label = self.data_set.load_label(example_idx)
            io_pairs.append(current_io_pair)
            labels.append(current_label)

        io_pairs_full = torch.cat(io_pairs, dim=0)
        labels_full = torch.cat(labels, dim=0)

        return io_pairs_full, labels_full
